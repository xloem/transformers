# coding=utf-8
# Copyright 2022 Bo PENG and the RWKV Dev Team and The HuggingFace 
# Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch RWKV2 model. """

import copy
import json
import numpy as np
import math
import os
from pathlib import Path
import sys
import time
import types

import torch
import torch.utils.checkpoint
from packaging import version

from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from typing import Optional, Tuple, Union

from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    CausalLMOutput,
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import logging
from .configuration_rwkv2 import RWKV2Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "rwkv-v2"
_CONFIG_FOR_DOC = "RWKV2Config"
_TOKENIZER_FOR_DOC = "PreTrainedTokenizerFast"

RWKV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    #"rwkv-v2-dagw-l6",
    # See all RWKV2 models at https://huggingface.co/models?filter=rwkv2
]

def load_cuda_kernels(t_max = 1024, b_group_forward = 8, b_group_backward = 2):
    global timex_cuda
    try:
        from torch.utils.cpp_extension import load

        def append_root(files):
            src_folder = os.path.dirname(os.path.realpath(__file__))
            return [Path(src_folder, file) for file in files]

        src_files = append_root(
            ["cuda/timex_op.cpp", "cuda/timex_cuda.cu"]
        )

        timex_cuda = load(
            src_files, 
            name="timex", 
            verbose=True, 
            extra_cuda_cflags=['--use_fast_math', '--extra-device-vectorization', f'-DTmax={T_MAX}', f'-DBF={B_GROUP_FORWARD}', f'-DBB={B_GROUP_BACKWARD}',],
        )

        return True
    except Exception:
        timex_cuda = None
        return False


class TimeX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, k, B, C, T, eps):
        ctx.B = B
        ctx.C = C
        ctx.T = T
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0
        w = w.contiguous()
        k = k.contiguous()
        ctx.save_for_backward(w, k)
        wk = torch.empty((B, C, T), device='cuda',
                        memory_format=torch.contiguous_format)
        timex_cuda.forward(w, k, wk, eps, B, C, T)
        return wk

    @staticmethod
    def backward(ctx, gwk):
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0
        w, k = ctx.saved_tensors
        gw = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                        memory_format=torch.contiguous_format)
        gk = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                        memory_format=torch.contiguous_format)
        timex_cuda.backward(w, k, gwk.contiguous(), gw,
                            gk, ctx.B, ctx.C, ctx.T)
        return (gw.sum(dim=0), gk, None, None, None, None)


def load_tf_weights_in_rwkv2(model, config, tf_checkpoint_path):
    raise NotImplementedError('Not supported for RWKV2 models')


class RWKV2ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        with torch.no_grad(): # init to "shift half of the channels"
            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd // 2):
                x[0,0,i] = 0
        self.time_mix = nn.Parameter(x)

        hidden_sz = 4 * config.n_ffn
        self.key = nn.Linear(config.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, config.n_embd, bias=False)

        self.value.scale_init = 0 # init to zero for faster convergence (use your own initializer)
        self.receptance.scale_init = 0 # init to zero for faster convergence (use your own initializer)

    def forward(self, x):
        x = x * self.time_mix + self.time_shift(x) * (1 - self.time_mix)

        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        
        rkv = torch.sigmoid(self.receptance(x)) * kv
        return rkv


class RWKV2TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        attn_sz = config.n_attn
        attn_sz = int(config.n_head * ((attn_sz + config.n_head/2) // config.n_head))

        assert attn_sz % config.n_head == 0
        self.head_size = attn_sz // config.n_head

        f1_begin = 3.0
        f1_end = 1.2
        f2_begin = 0.65
        f2_end = 0.4
                    
        with torch.no_grad(): # initial time_w curves for better convergence
            decay_speed = torch.ones(config.n_head, 1)
            first_sa_layer_id = 1
            for h in range(config.n_head):
                f1 = f1_begin + (layer_id-first_sa_layer_id) / (config.n_layer-1-first_sa_layer_id) * (f1_end - f1_begin)
                f2 = f2_begin + (layer_id-first_sa_layer_id) / (config.n_layer-1-first_sa_layer_id) * (f2_end - f2_begin)
                if layer_id == first_sa_layer_id:
                    f1 += 0.5
                if layer_id == config.n_layer-2:
                    f2 = 0.4
                if layer_id == config.n_layer-1:
                    f2 = 0.37
                decay_speed[h][0] = math.pow(f2, h / (config.n_head-1) * 7) * f1
        self.time_decay = nn.Parameter(torch.log(decay_speed))

        self.time_curve = torch.tensor([-(config.ctx_len - 2 - i) for i in range(config.ctx_len-1)]).unsqueeze(0)
        if RUN_DEVICE == 'cuda':
            self.time_curve = self.time_curve.to('cuda')
        self.time_first = nn.Parameter(torch.ones(config.n_head, 1) * math.log(0.3))

        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        with torch.no_grad():
            ww = torch.ones(1,1,config.n_embd)
            for i in range(config.n_embd // 2):
                ww[0,0,i] = 0
        self.time_mix = nn.Parameter(ww)

        self.key = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, attn_sz, bias=False)

        self.output = nn.Linear(attn_sz, config.n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0


class RWKV2Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        self.att = RWKV2TimeMix(config, layer_id)
        self.ffn = RWKV2ChannelMix(config, layer_id)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.att(x)
        x = self.ln2(x)
        x = x + self.ffn(x)
        return x


class RWKV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = RWKV2Config
    load_tf_weights = load_tf_weights_in_rwkv2
    base_model_prefix = "rwkv2"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RWKV2Encoder):
            module.gradient_checkpointing = value


RWKV2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config ([`~RWKV2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RWKV2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`RWKV2Tokenizer`].
            See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare RWKV2 transformer outputting raw hidden-states without any specific head on top.",
    RWKV2_START_DOCSTRING,
)
class RWKV2Model(RWKV2PreTrainedModel):
    """
    An implementation of the RWKV v2 model, see https://github.com/BlinkDL/RWKV-LM
    
    """

    def __init__(self, config):
        super().__init__()
        self.step = 0
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.Sequential(*[RWKV2Block(config, i) for i in range(config.n_layer)])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.ctx_len = config.ctx_len

        # Initialize weights and apply final processing
        self.post_init()

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def post_init(self): # fancy initialization of all lin & emb layer in the module
        super().post_init()

        for m in module.modules():
            if not isinstance(m, (nn.Linear, nn.Embedding)):
                continue
            with torch.no_grad():
                name = '[unknown weight]'
                for name, parameter in module.named_parameters(): # find the name of the weight
                    if id(m.weight) == id(parameter):
                        break

                shape = m.weight.data.shape
                gain = 1.0
                scale = 1.0 # extra scale for gain

                if isinstance(m, nn.Embedding):
                    gain = math.sqrt(max(shape[0], shape[1]))
                    if shape[0] == self.config.vocab_size and shape[1] == config.n_embd: # token emb?
                        scale = 1e-4
                    else:
                        scale = 0

                if isinstance(m, nn.Linear):
                    if m.bias is not None:
                        m.bias.data.zero_()
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])
                    if shape[0] == self.config.vocab_size and shape[1] == config.n_embd: # final projection?
                        scale = 0.5

                if hasattr(m, 'scale_init'):
                    scale = m.scale_init

                # print(str(shape[0]).ljust(5), str(shape[1]).ljust(5), f'{round(scale,2):g}'.ljust(4), name)

                gain *= scale
                if scale == -999:
                    nn.init.eye_(m.weight)
                elif gain == 0:
                    nn.init.zeros_(m.weight) # zero init is great for some RWKV matrices
                elif gain > 0:
                    nn.init.orthogonal_(m.weight, gain=gain)
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=-scale)


    def get_optimizer_groups(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        # return them as a set of groups to be passed to Adam, e.g. 
        #
        # optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)

        decay = set()
        no_decay = set()

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def get_input_embeddings(self):
        raise NotImplementedError
        # return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        raise NotImplementedError
        # self.embeddings.word_embeddings = value

    def get_ctx_len(self):
        return self.ctx_len

    @add_start_docstrings_to_model_forward(RWKV2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape `(batch_size, 1)`
            instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up
            decoding (see `past_key_values`).
        """

    def forward(self, idx, targets=None):
        self.step += 1
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."
        embeddings = self.emb(idx)

        hidden_states = self.h(embeddings)

        output_layer = self.ln_out(hidden_states)
        
        head = self.head(output_layer)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(head.view(-1, head.size(-1)), targets.view(-1))

        return head, loss



@add_start_docstrings(
    """RWKV2 Model with a `language modeling` head on top for CLM fine-tuning. """, RWKV2_START_DOCSTRING
)
class RWKV2ForCausalLM(RWKV2PreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `RWKV2ForCausalLM` as a standalone, add `is_decoder=True.`")

        self.rwkv2 = RWKV2Model(config)
        self.cls = RWKV2OnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(RWKV2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2
            tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
            tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two
            additional tensors are only required when the model is used as a decoder in a Sequence to Sequence
            model.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
            cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential
            decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape `(batch_size, 1)`
            instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up
            decoding (see `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import RWKV2Tokenizer, RWKV2ForCausalLM, RWKV2Config
        >>> import torch

        >>> tokenizer = RWKV2Tokenizer.from_pretrained('rwkv-v2')
        >>> config = RWKV2Config.from_pretrained("rwkv-v2")
        >>> config.is_decoder = True
        >>> model = RWKV2ForCausalLM.from_pretrained('rwkv-v2', config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.rwkv2(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],)
        return reordered_past
