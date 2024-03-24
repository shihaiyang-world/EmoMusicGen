import torch
from fast_transformers.builders import RecurrentDecoderBuilder
from fast_transformers.masking import TriangularCausalMask

# 首先，用RecurrentDecoderBuilder创建decoder
decoder_builder = RecurrentDecoderBuilder()
decoder = decoder_builder.get()

# 假定我们已经得到了encoder的输出context
context = encoder_output # Encoder的上下文表示 [batch_size, context_length, features]

# 为了生成序列，我们需要始终维护加入了最新token的序列
generated_sequence = []  # 保存已生成的token
input_sequence = torch.tensor([[start_token]])  # 某个起始token [batch_size, 1]

# 初始化隐藏状态为None
# 注意: 隐藏状态可以是None或者包含以往解码器隐藏层状态的元组
hidden_state = None

# 假设有一个函数通过某种方式从decoder的输出中选择下一个token (比如softmax + argmax)
# 如：choose_next_token(logits)
choose_next_token = ...

for _ in range(1):
    # 使用单步解码更新hidden_state以及产生logits
    logits, hidden_state = decoder(input_sequence, context, hidden_state)
    next_token = choose_next_token(logits[:, -1])
    generated_sequence.append(next_token.item())

    # 更新input_sequence以包括刚生成的token
    next_token = next_token.view(1, 1)  # 维度可能需要调整以匹配 [batch_size, 1]
    input_sequence = torch.cat((input_sequence, next_token), dim=1)

    # 创建适用于新input_sequence的causal mask
    causal_mask = TriangularCausalMask(input_sequence.size(1))

    # 检查是否结束序列生成（比如检查是否生成了end-of-sequence token）
    if next_token == eos_token:
        break

# 最终，generated_sequence 包含了生成的序列的token