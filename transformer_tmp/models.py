import torch.nn as nn
import os
from os.path import exists
import torch
import torch.nn as nn
# Pad: 用于对句子进行长度填充，官方地址为：https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
from torch.nn.functional import log_softmax, pad
import math
# copy: 用于对模型进行深拷贝
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
# 和matplotlib.pypolt类似，用于绘制统计图，但功能功能强大
# 可以绘制可交互的统计图。官网地址为：https://altair-viz.github.io/getting_started/overview.html
# import altair as alt
# 用于将 iterable 风格的 dataset 转为 map风格的 dataset，详情可参考：https://blog.csdn.net/zhaohongfei_358/article/details/122742656
# from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
# 用于构建词典
# from torchtext.vocab import build_vocab_from_iterator
# datasets：用于加载Multi30k数据集
# import torchtext.datasets as datasets
# spacy: 一个易用的分词工具，详情可参考：https://blog.csdn.net/zhaohongfei_358/article/details/125469155
import spacy
# GPU工具类，本文中用于显示GPU使用情况
# import GPUtil
# 用于忽略警告日志
import warnings

# 设置忽略警告
warnings.filterwarnings("ignore")


class EncoderDecoder(nn.Module):
    """
    一个标准的EncoderDecoder模型。在本教程中，这么类就是Transformer
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        encoder: Encoder类对象。Transformer的Encoder
        decoder： Decoder类对象。 Transformer的Decoder
        src_embed: Embeddings类对象。 Transformer的Embedding和Position Encoding
                   负责对输入inputs进行Embedding和位置编码
        tgt_embed: Embeddings类对象。 Transformer的Embedding和Position Encoding
                   负责对“输入output”进行Embedding和位置编码
        generator: Generator类对象，负责对Decoder的输出做最后的预测（Linear+Softmax）
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        src: 未进行word embedding的句子，例如`[[ 0, 5, 4, 6, 1, 2, 2 ]]`
             上例shape为(1, 7)，即batch size为1，句子词数为7。其中0为bos，
             1为eos, 2为pad

        tgt: 未进行word embedding的目标句子，例如`[[ 0, 7, 6, 8, 1, 2, 2 ]]`

        src_mask: Attention层要用的mask，对于source来说，就是要盖住非句子的部分，
                  例如`[[True,True,True,True,True,False,False]]`。相当于对上面
                  `[[ 0, 5, 4, 6, 1, 2, 2 ]]`中最后的`2,2`进行掩盖。

        tgt_mask: Decoder的Mask Attention层要用的。该shape为(N, L, L)，其中N为
                  batch size, L为target的句子长度。例如(1, 7, 7)，对于上面的例子，
                  值为：
                  [True,False,False,False,False,False,False], # 从该行开始，每次多一个True
                  [True,True,False,False,False,False,False],
                  [True,True,True,False,False,False,False],
                  [True,True,True,True,False,False,False],
                  [True,True,True,True,True,False,False], # 由于该句一共5个词，所以从该行开始一直都只是5个True
                  [True,True,True,True,True,False,False],
                  [True,True,True,True,True,False,False],
        """

        # 注意，这里的返回是Decoder的输出，并不是Generator的输出，因为在EncoderDecoder
        # 的forward中并没有使用generator。generator的调用是放在模型外面的。
        # 这么做的原因可能是因为Generator不算是Transformer的一部分，它只能算是后续处理
        # 分开来的话也比较方便做迁移学习。另一方面，推理时只会使用输出的最后一个tensor送给
        # generator，而训练时会将所有输出都送给generator。
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        该encode做三件事情：
        1. 对src进行word embedding
        2. 将word embedding与position encoding相加
        3. 通过Transformer的多个encoder层计算结果，输出结果称为memory
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        memory: Transformer Encoder的输出

        decoder和encoder执行过程基本一样：
        1. 对src进行word embedding
        2. 将word embedding与position encoding相加
        3. 通过Transformer的多个decoder层计算结果

        当完成decoder的计算后，接下来可以使用self.generator（nn.Linear+Softmax）来进行最后的预测。
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    Decoder的输出会送到Generator中做最后的预测。
    """

    def __init__(self, d_model, vocab):
        """
        d_model: dimension of model. 这个值其实就是word embedding的维度。
                 例如，你把一个词编码成512维的向量，那么d_model就是512
        vocab: 词典的大小。
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        x为Decoder的输出，例如x.shape为(1, 7, 128)，其中1为batch size, 7为句子长度，128为词向量的维度

        这里使用的是log_softmax而非softmax，效果应该是一样的。
        据说log_softmax能够解决函数overflow和underflow，加快运算速度，提高数据稳定性。
        可以参考：https://www.zhihu.com/question/358069078/answer/912691444
        该作者说可以把softmax都可以尝试换成log_softmax
        """
        return log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    """
    该方法负责产生多个相同的层。
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    Encoder的核心部分就是多个相同的EncoderLayer堆叠而成。
    """

    def __init__(self, layer, N):
        """
        初始化传入两个参数：
        layer: 要堆叠的层，对应下面的EncoderLayer类
        N: 堆叠多少次
        """
        super(Encoder, self).__init__()
        # 将Layer克隆N份
        self.layers = clones(layer, N)
        # LayerNorm层就是BatchNorm。也就是对应Transformer中
        # “Add & Norm”中的“Norm”部分
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        x: 进行过Embedding和位置编码后的输入inputs。Shape为(batch_size, 词数，词向量维度)
           例如(1, 7, 128)，batch_size为1，7个词，每个词128维
        mask: src_mask，请参考EncoderDecoder.forward中的src_mask注释
        """

        # 一层一层的执行，前一个EncoderLayer的输出作为下一层的输入
        for layer in self.layers:
            x = layer(x, mask)

        # 你可能会有疑问，为什么这里会有一个self.norm(x)，
        # 这个疑问会在后面的`SublayerConnection`中给出解释
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    Norm层，其实该层的作用就是BatchNorm。与`torch.nn.BatchNorm2d`的作用一致。
    torch.nn.BatchNorm2d的官方文档地址：https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

    该LayerNorm就对应原图中 “Add & Norm”中“Norm”的部分
    """

    def __init__(self, features, eps=1e-6):
        """
        features: int类型，含义为特征数。也就是一个词向量的维度，例如128。该值一般和d_model一致。
        """
        super(LayerNorm, self).__init__()
        """
        这两个参数是BatchNorm的参数，a_2对应gamma(γ), b_2对应beta(β)。
        而nn.Parameter的作用就是将这个两个参数作为模型参数，之后要进行梯度下降。
        """
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # epsilon，一个很小的数，防止分母为0
        self.eps = eps

    def forward(self, x):
        """
        x： 为Attention层或者Feed Forward层的输出。Shape和Encoder的输入一样。（其实整个过程中，x的shape都不会发生改变）。
            例如，x的shape为(1, 7, 128)，即batch_size为1，7个单词，每个单词是128维度的向量。
        """

        # 按最后一个维度求均值。mean的shape为 (1, 7, 1)
        mean = x.mean(-1, keepdim=True)
        # 按最后一个维度求方差。std的shape为 (1, 7, 1)
        std = x.std(-1, keepdim=True)
        # 进行归一化，详情可查阅BatchNorm相关资料。
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    在一个Norm层后跟一个残差连接。
    注意，为了代码简洁，Norm层是在最前面，而不是在最后面
    """

    def __init__(self, size, dropout):
        """
        这里的size就是d_model。也就是词向量的维度
        """
        super(SublayerConnection, self).__init__()
        # BatchNorm层
        self.norm = LayerNorm(size)
        # BatchNorm的dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        x：本层的输入，前一层的输出。
        sublayer: 这个Sublayer就是Attention层或Feed ForWard层。
        """
        return x + self.dropout(sublayer(self.norm(x)))



