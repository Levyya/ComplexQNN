# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  models.py.py
@Time    :  2022/11/21 9:51
@Author  :  Levy
@Contact :  1091697485@qq.com
@Desc    :  None
"""
from typing import Optional, Tuple
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Linear, Conv1d
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn import Activation
from allennlp.nn.util import min_value_of_dtype
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexReLU
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
from .utils import mine_complex_max_pool2d, abs_fn


class ComplexNet(nn.Module):

    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 10, 5, 1)
        self.bn = ComplexBatchNorm2d(10)
        self.conv2 = ComplexConv2d(10, 20, 5, 1)
        self.fc1 = ComplexLinear(4 * 4 * 20, 500)
        self.fc2 = ComplexLinear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        # torch.Size([64, 10, 24, 24])
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        # torch.Size([64, 10, 12, 12])
        x = self.bn(x)
        x = self.conv2(x)
        # torch.Size([64, 20, 8, 8])
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        # torch.Size([64, 20, 4, 4])
        x = x.view(-1, 4 * 4 * 20)
        # torch.Size([64, 320])
        x = self.fc1(x)
        # torch.Size([64, 500])
        x = complex_relu(x)
        x = self.fc2(x)
        # torch.Size([64, 10])
        x = x.abs()
        x = F.log_softmax(x, dim=1)
        # torch.Size([64, 10])
        return x


class TextCNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, out_channels=100, kernel_size=[3, 4, 5]):
        super(TextCNN, self).__init__()
        self.out_channels = out_channels

        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 2. Convs
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.out_channels, (kernel_size_, embedding_dim)) for kernel_size_ in kernel_size])
        # 3. Measure
        self.fc1 = self.fc2 = nn.Linear(3 * self.out_channels, hidden_size)
        self.fc2 = self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # torch.Size([64, 128])
        embedding = self.embedding(x).unsqueeze(1)
        # torch.Size([64, 1, 128, 768])
        convs = [torch.tanh(conv(embedding)).squeeze(3) for conv in self.convs]
        # [[64, 100, 126], [64, 100, 125], [64, 100, 124]]
        pool_out = [F.max_pool1d(block, 1, block.shape[-1]).squeeze() for block in convs]
        # [[64, 100] * 3]
        pool_out = torch.cat(pool_out, dim=1)
        out = self.fc1(self.dropout(pool_out))
        out = self.fc2(self.dropout(out))
        logits = F.log_softmax(out, dim=1)
        return logits


class ComplexTextCNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, out_channels=100, kernel_size=[3, 4, 5]):
        super(ComplexTextCNN, self).__init__()
        self.out_channels = out_channels

        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 2. Convs
        self.convs = nn.ModuleList(
            [ComplexConv2d(1, self.out_channels, (kernel_size_, embedding_dim)) for kernel_size_ in kernel_size])
        # 3. Measure
        self.fc1 = self.fc2 = ComplexLinear(3 * self.out_channels, hidden_size)
        self.fc2 = self.fc2 = ComplexLinear(hidden_size, num_classes)

    def forward(self, x):
        # torch.Size([64, 128])
        embedding = self.embedding(x).unsqueeze(1).type(torch.complex64)
        # torch.Size([64, 1, 128, 768])
        convs = [complex_relu(conv(embedding)).squeeze(3) for conv in self.convs]
        # [[64, 100, 126], [64, 100, 125], [64, 100, 124]]
        pool_out = [mine_complex_max_pool2d(block.unsqueeze(-1), 1, block.shape[-1]).squeeze() for block in convs]
        # [[64, 100] * 3]
        pool_out = torch.cat(pool_out, dim=1)
        logits = self.fc1(pool_out)
        logits = self.fc2(logits)
        logits = F.log_softmax(abs_fn(logits), dim=1)
        return logits


@Seq2VecEncoder.register("complexcnn")
class ComplexConvSeq2Vec(Seq2VecEncoder):

    def __init__(
            self,
            embedding_dim: int,
            num_filters: int,
            ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),
            conv_layer_activation: Activation = None,
            output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = ComplexReLU()
        # self._activation = conv_layer_activation or Activation.by_name("relu")()

        self._convolution_layers = nn.ModuleList([
            ComplexConv2d(1, self._num_filters, (ngram_size, self._embedding_dim))
            for ngram_size in self._ngram_filter_sizes
        ])
        # self._convolution_layers = [
        #     Conv1d(
        #         in_channels=self._embedding_dim,
        #         out_channels=self._num_filters,
        #         kernel_size=self._ngram_filter_sizes,
        #     )
        #     for ngram_size in self._ngram_filter_sizes
        # ]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module("conv_layer_%d" % i, conv_layer)

        maxpool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
        if output_dim:
            self.projection_layer = ComplexLinear(maxpool_output_dim, output_dim)
            self._output_dim = output_dim
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1)
        else:
            # If mask doesn't exist create one of shape (batch_size, num_tokens)
            mask = torch.ones(tokens.shape[0], tokens.shape[1], device=tokens.device).bool()
        tokens = tokens.unsqueeze(1).type(torch.complex64)
        mask = mask.type(torch.complex64)
        # torch.Size([batch_size, num_tokens])
        # embedding = self.embedding(tokens).unsqueeze(1).type(torch.complex64)
        # torch.Size([batch_size, 1, num_tokens, embedding_dim])
        convs = [
            # self._activation(conv(tokens)).squeeze(3)
            conv(tokens).squeeze(3)
            for conv in self._convolution_layers
        ]
        # [[64, 100, 126], [64, 100, 125], [64, 100, 124]]
        # maxpool_output = [mine_complex_max_pool2d(block.unsqueeze(-1), 1, block.shape[-1]).squeeze() for block in convs]
        # maxpool
        
        # [[64, 100] * 3]
        pool_out = torch.cat(maxpool_output, dim=1)
        if self.projection_layer:
            # result = abs_fn(self.projection_layer(pool_out))
            result = self.projection_layer(pool_out).real
        else:
            result = pool_out
        return result

        # filter_outputs = []
        # batch_size = tokens.shape[0]
        # last_unmasked_tokens = mask.sum(dim=1).unsqueeze(dim=-1)
        # for i in range(len(self._convolution_layers)):
        #     convolution_layer = getattr(self, "conv_layer_{}".format(i))
        #     pool_length = tokens.shape[2] - convolution_layer.kernel_size[0] + 1
        #
        #     # Forward pass of the convolutions.
        #     # shape: (batch_size, num_filters, pool_length)
        #     activations = self._activation(convolution_layer(tokens))
        #
        #     indices = (
        #         torch.arange(pool_length, device=activations.device)
        #         .unsqueeze(0)
        #         .expand(batch_size, pool_length)
        #     )
        #     activations_mask = indices.ge(
        #         last_unmasked_tokens - convolution_layer.kernel_size[0] + 1
        #     )
        #     # shape: (batch_size, num_filters, pool_length)
        #     activations_mask = activations_mask.unsqueeze(1).expand_as(activations)
        #     activations = activations + (activations_mask * min_value_of_dtype(activations.dtype))
        # #
        # if self.projection_layer:
        #     result = self.projection_layer(maxpool_output)
        # else:
        #     result = maxpool_output
        # return result


@Seq2VecEncoder.register("complexqnn")
class ComplexQNNSeq2Vec(Seq2VecEncoder):
    """
    # 此处可以考虑给tokens随机加上一个虚部，或者虚部使用另外一种embedding!!!
    # TODO
    # embedding_type: random, GloVe or Word2vec, contextual embedding
    # 思路：a + 1j * b
    # tokens = ptm_tokens + 1j * ptm_tokens
    # 实现：通过real_embedder将tokens -> real_part, 通过imag_encoder将tokens -> imag_part
    # final_encoder = real_part + 1j * imag_part
    # 注：这是一种简单的实现方式，另外还可以构造complex-valued embedding，但前者更符合模型构建的意义
    # 意义：复数词向量空间/希尔伯特空间能够表示更加复杂的意义（相比于经典实数空间），提高模型的异构性，
    # 尝试使用实部、虚部分别编码不同的信息，一方面不同信息可以相互纠缠，另一方面模型表达能力得到提升

    # encoder: complex_tokens -> vector
    # Process:
    # token: [batch_size, seq_len]
    # complex_embedder: ptm_tokens + 1j * ptm_tokens
    # token: [batch_size, seq_len, emb_dim]
    # 分两种方向：其一量子启发式模型，其二复数值神经网络
    # 其一：量子启发式模型
    # 前面token对应词语的量子态表示，通过外积、求和（加权），可以得到句子的密度矩阵表示
    # sequence: [batch_size, emb_dim, emb_dim]
    # 演化：学习文本内部特征，不改变维度 [emb_dim, emb_dim]；gru, cnn, fc, transformer
    # sequence: [batch_size, emb_dim, emb_dim]
    # 测量：tr(PV), 其中P是测量算子对应的密度矩阵
    # prediction: [batch_size, num_labels]
    # 其二：复数值神经网络
    # encoder: [seq2seq], seq2vec; 具体地，complexcnn, complexgru, complexdnn
    # sequence: [batch_size, hidden_size]
    # classifer: nn.Linear()
    # sequence: [batch_size, num_labels]

    # projection: ComplexLinear(num_filters * len(ngrams_filter_sizes), output_dim)
    # result = abs_fn(self.projection_layer(pool_out))
    """

    def __init__(
            self,
            embedding_dim: int,
            output_dim: Optional[int] = None,
            device: str = 'cuda:2'
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._activation = ComplexReLU()
        # self.gru = nn.GRU(self._embedding_dim, self._embedding_dim, 2)
        projector_real = nn.init.uniform_(torch.nn.Parameter(torch.Tensor(self._embedding_dim, 1), requires_grad=False)).to(device)
        projector_imag = nn.init.uniform_(torch.nn.Parameter(torch.Tensor(self._embedding_dim, 1), requires_grad=False)).to(device)
        projector = projector_real + 1j * projector_imag
        self.projector = projector
        self.cf1 = ComplexLinear(self._embedding_dim, self._embedding_dim)
        self.cf2 = ComplexLinear(self._embedding_dim, self._embedding_dim)
        if output_dim:
            self.projection_layer = ComplexLinear(self._embedding_dim, output_dim)
            self._output_dim = output_dim
        else:
            self.projection_layer = None
            self._output_dim = self._embedding_dim

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._output_dim
    
    def simple_measure(self, x: torch.Tensor):
        # x [batch, seq_len, emb_dim]
        pooler = torch.mean(x, dim=-2)  # [batch, emb_dim]
        return pooler.real
    
    def measure(self, x: torch.Tensor):
        # TODO: quantum measure
        # P = self.projector @ self.projector.permute(1, 0)
        # result = torch.cat([
        #     torch.diag(P @ batch).abs().unsqueeze(0)
        #     for batch in V
        # ]).to(tokens.device)
        # return result
        return x.real

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1)
        else:
            # If mask doesn't exist create one of shape (batch_size, num_tokens)
            mask = torch.ones(tokens.shape[0], tokens.shape[1], device=tokens.device).bool()
        # tokens = tokens.unsqueeze(1).type(torch.complex64)
        if tokens.dtype != torch.complex64:
            imag_tokens = torch.randn_like(tokens)
            tokens = tokens + 1j * imag_tokens  
        
        ## tokens: {shape=[batch, seq_len, emb_dim], dtype=torch.complex64}
        
        # Method1: Quantum-inspired model (origin)
        # step1：得到密度矩阵：外积 + 求和
        # tokens = tokens.unsqueeze(-1)
        # V = torch.mean(tokens @ tokens.permute(0, 1, 3, 2), dim=1)
        # # step2: evolution
        # V_r, _ = self.gru(V.real)
        # V_i, _ = self.gru(V.imag)
        # V = V_r + 1j * V_i
        # # density_matrix: [batch_size, emb_dim, emb_dim]
        # # step3: 测量
        # P = self.projector @ self.projector.permute(1, 0)
        # result = torch.cat([
        #     torch.diag(P @ batch).abs().unsqueeze(0)
        #     for batch in V
        # ]).to(tokens.device)
        # return result
        
        # Method2: Quantum-inspired model (complexnn)
        x = self.cf1(tokens)
        # x = self.cf2(tokens)
        result = self.simple_measure(x)
        return result