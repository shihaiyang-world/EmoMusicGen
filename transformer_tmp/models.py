'''
定一个最简单的Transformer，带入自己的模型，进行训练过程
'''

import torch
import torch.nn as nn
from public_layer import PositionalEncoding, MultiHeadAttention, network_paras
import numpy as np
import math
import utils
import torch.nn.functional as F

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
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, emb_sizes, n_class, dropout=0.1, d_ff=2048, is_training=True):
        super(TransformerModel, self).__init__()
        # d_ff 最后一层位置前馈网络中内层的维数
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_decoder_layers)])

        self.n_class = n_class
        self.emb_sizes = emb_sizes
        self.d_model = d_model
        self.is_training = is_training

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

    def forward(self, src, tgt, loss_mask=None):
        # 定义src_mask，即所有的词都是有效的，没有填充词
        # src_mask = torch.ones(src.size(0), src.size(1), src.size(2))
        src_embedded = self.common_embedding(src)
        enc_output_memory = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output_memory = enc_layer(enc_output_memory, None)

        # encoder层  src的输出作为decoder的输入
        tgt_mask = None
        if self.is_training:
            tgt_mask = self.subsequent_mask(tgt.size(1))
            tgt_mask = tgt_mask.cuda()
        tgt_embedded = self.common_embedding(tgt)

        # decoder层  tgt的embedding，encoder的输出作为memory作为输入
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output_memory, None, tgt_mask)


        # decoder 重构出来的结果，这个结果与真实目标target对比，产生loss
        if self.is_training:
            # training 计算loss返回
            y_tempo, y_chord, y_type, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion = self.forward_output(
                dec_output)
            # 计算loss的逻辑
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
            # 最后过一层softmax  因为是多个embedding在一起，用多个proj计算所有的loss  取平均
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
        else:
            # 预测生成时，返回的是最后一个时间步的结果，而不是loss
            y_tempo, y_chord, y_type, y_barbeat, y_pitch, y_duration, y_velocity, y_emotion = self.forward_output(
                dec_output)
            y_tempo = y_tempo[:, -1, :].unsqueeze(0)
            y_chord = y_chord[:, -1, :].unsqueeze(0)
            y_type = y_type[:, -1, :].unsqueeze(0)
            y_barbeat = y_barbeat[:, -1, :].unsqueeze(0)
            y_pitch = y_pitch[:, -1, :].unsqueeze(0)
            y_duration = y_duration[:, -1, :].unsqueeze(0)
            y_velocity = y_velocity[:, -1, :].unsqueeze(0)
            y_emotion = y_emotion[:, -1, :].unsqueeze(0)
            return nn.Softmax(dim=-1)(y_tempo), nn.Softmax(dim=-1)(y_chord), nn.Softmax(dim=-1)(y_type), nn.Softmax(dim=-1)(y_barbeat), nn.Softmax(dim=-1)(y_pitch), nn.Softmax(dim=-1)(y_duration), nn.Softmax(dim=-1)(y_velocity), nn.Softmax(dim=-1)(y_emotion)


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



    def generate_from_scratch(self, dictionary, emotion_tag, key_tag=None, n_token=8, display=True):
        event2word, word2event = dictionary

        classes = word2event.keys()

        def print_word_cp(cp):

            result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]

            for r in result:
                print('{:15s}'.format(str(r)), end=' | ')
            print('')

        generated_key = None

        target_emotion = [0, 0, 0, 1, 0, 0, 0, emotion_tag]

        init = np.array([
            target_emotion,  # emotion
            [0, 0, 1, 2, 0, 0, 0, 0]  # bar
        ])

        cnt_token = len(init)
        with torch.no_grad():
            final_res = []
            # encoder的结果
            memory = None
            h = None

            cnt_bar = 1
            init_t = torch.from_numpy(init).long().cuda()
            print('------ initiate ------')

            for step in range(init.shape[0]):
                print_word_cp(init[step, :])
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(init[step, :][None, ...])
                # 采样类型
                h, y_type, memory = self.forward(input_, memory, is_training=False)

            print('------ generate ------')
            while (True):
                # sample others  采样其他的，音高，音长，力度等
                next_arr, y_emotion = self.froward_output_sampling(h, y_type)
                if next_arr is None:
                    return None, None

                final_res.append(next_arr[None, ...])

                if display:
                    print('bar:', cnt_bar, end='  ==')
                    print_word_cp(next_arr)

                # forward
                input_ = torch.from_numpy(next_arr).long().cuda()
                input_ = input_.unsqueeze(0).unsqueeze(0)
                h, y_type, memory = self.forward_hidden(
                    input_, memory, is_training=False)

                # end of sequence
                if word2event['type'][next_arr[3]] == 'EOS':
                    break

                if word2event['bar-beat'][next_arr[2]] == 'Bar':
                    cnt_bar += 1

        print('\n--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)

        return final_res, generated_key

    def froward_output_sampling(self, h, y_type, is_training=False):
        '''
        for inference
        '''

        # sample type
        y_type_logit = y_type[0, :]  # token class size
        cur_word_type = utils.sampling(y_type_logit, p=0.90, is_training=is_training)  # int
        if cur_word_type is None:
            return None, None

        if is_training:
            # unsqueeze 升维  squeeze 降维，维度=1的维去掉
            type_word_t = cur_word_type.long().unsqueeze(0).unsqueeze(0)
        else:
            type_word_t = torch.from_numpy(
                np.array([cur_word_type])).long().cuda().unsqueeze(0)  # shape = (1,1)

        tf_skip_type = self.word_emb_type(type_word_t).squeeze(0)  # shape = (1, embd_size)

        # concat
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        # project other
        y_tempo = self.proj_tempo(y_)
        y_chord = self.proj_chord(y_)
        y_barbeat = self.proj_barbeat(y_)

        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)
        y_emotion = self.proj_emotion(y_)

        # sampling gen_cond
        cur_word_tempo = utils.sampling(y_tempo, t=1.2, p=0.9, is_training=is_training)
        cur_word_barbeat = utils.sampling(y_barbeat, t=1.2, is_training=is_training)
        cur_word_chord = utils.sampling(y_chord, p=0.99, is_training=is_training)
        cur_word_pitch = utils.sampling(y_pitch, p=0.9, is_training=is_training)
        cur_word_duration = utils.sampling(y_duration, t=2, p=0.9, is_training=is_training)
        cur_word_velocity = utils.sampling(y_velocity, t=5, is_training=is_training)

        curs = [
            cur_word_tempo,
            cur_word_chord,
            cur_word_barbeat,
            cur_word_pitch,
            cur_word_duration,
            cur_word_velocity
        ]

        if None in curs:
            return None, None

        if is_training:
            cur_word_emotion = torch.from_numpy(np.array([0])).long().cuda().squeeze(0)
            # collect
            next_arr = torch.tensor([
                cur_word_tempo,
                cur_word_chord,
                cur_word_barbeat,
                cur_word_type,
                cur_word_pitch,
                cur_word_duration,
                cur_word_velocity,
                cur_word_emotion
            ])

        else:
            cur_word_emotion = 0

            # collect
            next_arr = np.array([
                cur_word_tempo,
                cur_word_chord,
                cur_word_barbeat,
                cur_word_type,
                cur_word_pitch,
                cur_word_duration,
                cur_word_velocity,
                cur_word_emotion
            ])

        return next_arr, y_emotion


class Generator(nn.Module):
    """
    Decoder的输出会送到Generator中做最后的预测。
    """
    def __init__(self):
        """
        d_model: dimension of model. 这个值其实就是word embedding的维度。
                 例如，你把一个词编码成512维的向量，那么d_model就是512
        vocab: 词典的大小。
        """
        super(Generator, self).__init__()

    def forward(self, x):


        return F.softmax(self.proj(x), dim=-1)