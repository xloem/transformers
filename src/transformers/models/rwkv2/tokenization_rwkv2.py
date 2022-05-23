# coding=utf-8
# Copyright 2022 BlinkDL and RWKV Core Team and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for RWKV2."""

import json
from pathlib import Path
from typing import List, Optional

from tokenizers import ByteLevelBPETokenizer

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        #"rwkv-v2": "https://huggingface.co/rwkv-v2/resolve/main/vocab.txt",
        "rwkv-v2": "vocab.json"
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "rwkv-v2": 1024,
}

class RWKV2Tokenizer(PreTrainedTokenizer):
    """
    Construct a RWKV2 tokenizer. Based on character-level input.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids"]

    def __init__(
            self,
            vocab_file,
            eos_token="\n",
            unk_token="<|unk|>",
            **kwargs
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        super().__init__(
            eos_token=eos_token, 
            unk_token=unk_token, 
            **kwargs
            )

        """ Initialisation"""

        with open(vocab_file, encodeing="utf-16") as vocab_file:
            vocab_loaded = json.load(vocab_file.load)
        self.encoder = {v: int(k) for k,v in vocab_loaded.items()}
        self.decoder = {int(k): v for k,v in vocab_loaded.items()}
        if unk_token.content not in self.encoder:
            self.unk_token_id = max(self.decoder) + 1
            self.encoder[unk_token] = self.unk_token_id
            self.decoder[unk_token_id] = unk_token.content
        else:
            self.unk_token_id = self.encoder(unk_token.content)
        if eos_token.content not in self.encoder:
            self.eos_token_id = max(self.decoder) + 1
            self.encoder[eos_token] = self.eos_token_id
            self.decoder[eos_token_id] = eos_token.content
        else:
            self.eos_token_id = self.encoder(eos_token.content)
        assert len(self.encoder) == len(self.decoder)

    @property
    def vocab_size(self):
        """ Returns vocab size """
        return len(self.decoder)

    def get_vocab(self):
        """ Returns vocab as a dict """
        return self.encoder

    def _tokenize(self, text):
        """ Returns a tokenized string. """

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token not in self.encoder:
            return self.unk_token_id
        return self.encoder[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder[index]

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        for char in list(tokens):
            if char in self.encoder:
                yield self.encoder(char)
            else:
                yield self.unk_token_id

    def save_vocabulary(self, save_directory):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        with open(Path(save_directory, 'vocab.json'), "w", encoding="utf-16") as vocab_file:
            vocab_file.write(json.dumps(self.decoder, ensure_ascii=False))


    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RWKV2 sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.
        RWKV2 does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`:  List of zeros.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)
