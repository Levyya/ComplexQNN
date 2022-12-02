# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  data.py
@Time    :  2022/11/26 17:02
@Author  :  Levy
@Contact :  1091697485@qq.com
@Desc    :  None
"""
import logging
import csv
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from typing import *

from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("mydataset_reader")
class TextClassificationTextReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 delimiter: str = ',',
                 testing: bool = False,
                 max_sequence_length: int = None,
                 segment_sentences: bool = False,
                 lazy: bool = True) -> None:
        """
        文本分类任务的datasetreader,从csv获取数据,head指定text,label.如:
        label   text
        sad    i like it.
        :param tokenizer: 分词器
        :param token_indexers:
        :param delimiter:
        :param testing:
        :param max_sequence_length:
        :param lazy: ?
        """
        super().__init__()  # lazy 的问题！！！
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._delimiter = delimiter
        self.testing = testing
        self._max_sequence_length = max_sequence_length
        self._segment_sentences = segment_sentences

    # @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, "r") as lines:
            for line in lines:
                # print(line)
                text, sentiment = line.strip().split("\t")
                tokens = self._tokenizer.tokenize(text)
                if self._max_sequence_length:
                    tokens = tokens[: self._max_sequence_length]
                text_field = TextField(tokens, self._token_indexers)
                label_field = LabelField(sentiment)
                yield Instance({"tokens": text_field, "label": label_field})

    def _truncate(self, tokens):
        """
        truncate a set of tokens using the provided sequence length
        """
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[:self._max_sequence_length]
        return tokens
    
    def text_to_instance(  # type: ignore
        self, text: str, label: Union[str, int] = None
    ) -> Instance:
        """
        # Parameters

        text : `str`, required.
            The text to classify
        label : `str`, optional, (default = `None`).
            The label for this text.

        # Returns

        An `Instance` containing the following fields:
            - tokens (`TextField`) :
              The tokens in the sentence or phrase.
            - label (`LabelField`) :
              The label label of the sentence or phrase.
        """

        fields: Dict[str, Field] = {}
        if self._segment_sentences:
            sentences: List[Field] = []
            sentence_splits = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits:
                word_tokens = self._tokenizer.tokenize(sentence)
                if self._max_sequence_length is not None:
                    word_tokens = self._truncate(word_tokens)
                sentences.append(TextField(word_tokens))
            fields["tokens"] = ListField(sentences)
        else:
            tokens = self._tokenizer.tokenize(text)
            if self._max_sequence_length is not None:
                tokens = self._truncate(tokens)
            fields["tokens"] = TextField(tokens)
        if label is not None:
            fields["label"] = LabelField(label, skip_indexing=self._skip_label_indexing)
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        if self._segment_sentences:
            for text_field in instance.fields["tokens"]:  # type: ignore
                text_field._token_indexers = self._token_indexers
        else:
            instance.fields["tokens"]._token_indexers = self._token_indexers  # type: ignore
