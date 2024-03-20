'''
定一个最简单的Transformer，带入自己的模型，进行训练过程
'''

import torch
import torch.nn as nn
from public_layer import PositionalEncoding, MultiHeadAttention, network_paras
import numpy as np
import math
from fast_transformers.masking import TriangularCausalMask as TriangularCausalMask_local

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 对model的根号倍数进行缩放
        # 乘以 根号 d_model 其实就是起到了一个标准化的作用
        return self.lut(x) * math.sqrt(self.d_model)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, emb_sizes, n_class, dropout=0.1, d_ff=2048):
        super(TransformerModel, self).__init__()
        # d_ff 最后一层位置前馈网络中内层的维数
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_decoder_layers)])

        self.n_class = n_class
        self.emb_sizes = emb_sizes
        self.d_model = d_model

        self.word_emb_tempo = Embeddings(self.n_class[0], self.emb_sizes[0])
        self.word_emb_chord = Embeddings(self.n_class[1], self.emb_sizes[1])
        self.word_emb_barbeat = Embeddings(self.n_class[2], self.emb_sizes[2])
        self.word_emb_type = Embeddings(self.n_class[3], self.emb_sizes[3])
        self.word_emb_pitch = Embeddings(self.n_class[4], self.emb_sizes[4])
        self.word_emb_duration = Embeddings(self.n_class[5], self.emb_sizes[5])
        self.word_emb_velocity = Embeddings(self.n_class[6], self.emb_sizes[6])
        self.word_emb_emotion = Embeddings(self.n_class[7], self.emb_sizes[7])

        self.linear = nn.Linear(np.sum(emb_sizes), d_model)
        self.posAtten = PositionalEncoding(d_model)

        self.loss_func = nn.CrossEntropyLoss(reduction='none')


        # individual output
        self.proj_tempo = nn.Linear(self.d_model, self.n_class[0])
        self.proj_chord = nn.Linear(self.d_model, self.n_class[1])
        self.proj_barbeat = nn.Linear(self.d_model, self.n_class[2])
        self.proj_type = nn.Linear(self.d_model, self.n_class[3])
        self.proj_pitch = nn.Linear(self.d_model, self.n_class[4])
        self.proj_duration = nn.Linear(self.d_model, self.n_class[5])
        self.proj_velocity = nn.Linear(self.d_model, self.n_class[6])
        self.proj_emotion = nn.Linear(self.d_model, self.n_class[7])

    def subsequent_mask(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        return subsequent_mask

    def forward(self, src, tgt, loss_mask):
        # 定义src_mask，即所有的词都是有效的，没有填充词
        # src_mask = torch.ones(src.size(0), src.size(1), src.size(2))
        tgt_mask = self.subsequent_mask(tgt.size(1))
        tgt_mask = tgt_mask.cuda()
        src_embedded = self.common_embedding(src)
        tgt_embedded = self.common_embedding(tgt)
        # encoder层  src的输出作为decoder的输入
        enc_output_memory = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output_memory = enc_layer(enc_output_memory, None)

        # decoder层  tgt的embedding，encoder的输出作为memory作为输入
        # mask怎么造？
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output_memory, None, tgt_mask)

        # 最后过一层softmax  因为是多个embedding在一起，用多个proj计算所有的loss
        # output = self.fc(dec_output)

        # decoder 重构出来的结果，这个结果与真实目标target对比，产生loss
        y_tempo, y_chord, y_type, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion = self.forward_output(dec_output)

        # reshape (b, s, f) -> (b, f, s)
        y_tempo = y_tempo[:, ...].permute(0, 2, 1)
        y_chord = y_chord[:, ...].permute(0, 2, 1)
        y_barbeat = y_barbeat[:, ...].permute(0, 2, 1)
        y_type = y_type[:, ...].permute(0, 2, 1)
        y_pitch = y_pitch[:, ...].permute(0, 2, 1)
        y_duration = y_duration[:, ...].permute(0, 2, 1)
        y_velocity = y_velocity[:, ...].permute(0, 2, 1)
        y_emotion = y_emotion[:, ...].permute(0, 2, 1)

        # loss
        loss_tempo = self.compute_loss(y_tempo, tgt[..., 0], loss_mask)
        loss_chord = self.compute_loss(y_chord, tgt[..., 1], loss_mask)
        loss_barbeat = self.compute_loss(y_barbeat, tgt[..., 2], loss_mask)
        loss_type = self.compute_loss(y_type, tgt[..., 3], loss_mask)
        loss_pitch = self.compute_loss(y_pitch, tgt[..., 4], loss_mask)
        loss_duration = self.compute_loss(y_duration, tgt[..., 5], loss_mask)
        loss_velocity = self.compute_loss(y_velocity, tgt[..., 6], loss_mask)
        loss_emotion = self.compute_loss(y_emotion, tgt[..., 7], loss_mask)

        # 返回各个Loss
        return loss_tempo, loss_chord, loss_barbeat, loss_type, loss_pitch, loss_duration, loss_velocity, loss_emotion

    def forward_output(self, decoder_outputs):
        # project other  沿着最后一维堆叠
        # individual output  做了一个全连接
        y_tempo = self.proj_tempo(decoder_outputs)
        y_chord = self.proj_chord(decoder_outputs)
        y_type = self.proj_type(decoder_outputs)
        y_barbeat = self.proj_barbeat(decoder_outputs)
        y_pitch = self.proj_pitch(decoder_outputs)
        y_duration = self.proj_duration(decoder_outputs)
        y_velocity = self.proj_velocity(decoder_outputs)
        y_emotion = self.proj_emotion(decoder_outputs)

        return y_tempo, y_chord, y_type, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion

    def compute_loss(self, predict, target, loss_mask=None):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss


    def common_embedding(self, x):
        # Embedding
        emb_tempo = self.word_emb_tempo(x[..., 0])
        emb_chord = self.word_emb_chord(x[..., 1])
        emb_barbeat = self.word_emb_barbeat(x[..., 2])
        emb_type = self.word_emb_type(x[..., 3])
        emb_pitch = self.word_emb_pitch(x[..., 4])
        emb_duration = self.word_emb_duration(x[..., 5])
        emb_velocity = self.word_emb_velocity(x[..., 6])
        emb_emotion = self.word_emb_emotion(x[..., 7])

        # 把特征concat  shape torch.Size([32, 1024, 1376])
        embs = torch.cat(
            [emb_tempo, emb_chord, emb_barbeat, emb_type, emb_pitch, emb_duration, emb_velocity, emb_emotion], dim=-1)
        # embs_ shape torch.Size([32, 1024, 512])
        embs_ = self.linear(embs)
        # encoding_input shape torch.Size([32, 1024, 512])
        encoding_input = self.posAtten(embs_)
        return encoding_input





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