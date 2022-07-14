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
""" RWKV2 model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

RWKV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "rwkv2": "https://huggingface.co/rwkv-v2/resolve/main/config.json",
    # See all RWKV2 models at https://huggingface.co/models?filter=rwkv2
}


class RWKV2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`~RWKV2Model`].
    It is used to instantiate an RWKV2 model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the RWKV2 [rwkv-v2](https://huggingface.co/rwkv-v2) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the RWKV2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~RWKV2Model`] or
            [`~TFRWKV2Model`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"sigmoid"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import RWKV2Model, RWKV2Config

    >>> # Initializing a RWKV2 rwkv-v2 style configuration
    >>> configuration = RWKV2Config()

    >>> # Initializing a model from the rwkv-v2 style configuration
    >>> model = RWKV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    Args:
        vocab_size = 50277
            Vocabulary size of the model
        ctx_len = 768
            Context length.
        n_layer = 12
            Number of layers
        n_embd = 768
            Use at least 1024 for BPE-level English
        time_decay
        time_first
    """

    model_type = "rwkv2"
    attribute_map = {
        "hidden_size": "n_embd",
        "num_hidden_layers": "n_layer",
    }   

    def __init__(
        self,
        vocab_size = 50277,
        ctx_len=768,
        n_layer=12,
        n_embd=768,
        n_head=768,
        n_attn=768,
        n_ffn=768,
#        hidden_act="sigmoid",
        is_encoder_decoder=False,
        use_cache=True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_attn = n_attn
        self.n_ffn = n_embd
#        self.hidden_act = hidden_act
        self.use_cache = use_cache

        super().__init__(
            **kwargs
        )

    @property
    def num_attention_heads(self):
        logger.info(f"The number of attention heads is the same as the model’s feature dimension.")
        return -1

    @num_attention_heads.setter
    def num_attention_heads(self, value):
        # Message copied from "An Attention Free Transformer"
        raise NotImplementedError(
            f"The number of attention heads is the same as the model’s feature dimension."
        )
