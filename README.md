## 基于情绪生成钢琴乐


有几个需要做的
1. 选择带有情绪的MIDI数据集   ✅
2. 预处理MIDI数据集    ✅  co-representation中已经处理好了
3. 训练模型      
   1. 先用**Transformer**能复现出来；  **embedding**  **encoder**  **decoder**  attention   ✅
   2. 加入CVAE模型   encoder decoder reparameterization  kl_loss  
   3. 加入情绪  用情绪标签来训练模型
4. 生成音乐   旋律文件转MIDI  或者直接生成MIDI ✅
   1. 加入情绪  胶囊？  把一段脑电识别后，保存情绪胶囊，然后生成音乐的时候，加入情绪胶囊 （常规的操作是根据情绪条件，效价唤醒的正负来影响音乐，我们认为这样选择太少，我们基于情绪胶囊来选择音乐，原生带有概率，信息更加丰富）
   2. 
5. 保存转wav  ✅这个已经知道如何做


开搞。


### 选择数据集  

目前根据读到的，目前观察有情绪的钢琴数据集。

Dataset: [EMOPIA: A Multi-Modal Pop Piano Dataset For Emotion Recognition and Emotion-based Music Generation](https://arxiv.org/abs/2108.01374) 
[Code](https://github.com/annahung31/EMOPIA?tab=readme-ov-file)  
[Demo](https://annahung31.github.io/EMOPIA/)

用于情感识别和基于情感的音乐生成的多模态流行钢琴数据集

1700个带情绪标签的音乐片段，情绪标签是按罗素四象限来的。Q1-Q4都有。

### 预处理数据集

`EMOPIA`的预处理代码

参考其他的论文中的代码：


```python
# midi2couper

 

```
术语：
velocity: 速率
pitch: 音高
duration: 持续时间
note: 音符
track: 轨道
pedal: 踏板
chord: 和弦
tempo: 速度
rest: 休止


### 训练网络

跑了一个LSTM生成的简单模型，大概思路就是通过把节奏数据集切分成一段一段的旋律，然后每一段旋律作为input,下一个音符作为target，
输入到LSTM中训练模型。

训练完成的模型，再根据步数n，预测出n步的旋律音符。根据预测出的n步旋律再生成MIDI文件。就完成了音乐生成。

此时有几个疑问，如何加入情绪？如何让生成的音乐更加有情感？

我觉得首先得跑通一个transformer的模型，复现一下，然后再考虑加入情绪。
最好是基于MIDI的，看transformer下，下一个音符生成的逻辑是什么。


这里可以参考的是：
YouTube上的一个UP，The Sound of AI公司创始人的分享。
包括LSTM生成音乐的实现也是他的分享。
[Code:基于Transformer生成旋律](https://github.com/musikalkemist/generativemusicaicourse)


最小MVP版本
先用四象限的方式生成音乐。先实现，再考虑优化。

实现方式：
用带情绪的MIDI数据集，训练一个transformer模型。直接训练一个模型出来。
```python
# Emotion:(1-4) 4象限
# Melody: 旋律  
# Tokenizer之后，输入到transformer模型中，输出下一个音符。

# 多大模型？
这样就简单了，现在最重要先把带情绪的旋律模型训练出来。能够根据情绪象限生成旋律。不用考虑情绪胶囊先。
```

文本怎么做情感分析的？音符能否借鉴一下对应的方式 构建Transformer模型。
加条件，VV在Transformer结尾提到了可以探索的地方，就是在生成时在metadata中加入Condition
Condition melody generation on metadata: 用metadata来生成旋律。
在编码器中传递。
元编码器，文本、或者艺术风格、或者情感。
在解码器中调节生成的音乐。

一个思路[在Transformer的Input加入Metadata](https://www.tdcommons.org/cgi/viewcontent.cgi?article=7612&context=dpubs_series)

1. adding a metadata embedding layer  元数据嵌入层
2. conditioning self-attention on the metadat
3. conditioning with gated self-attention 用门控的自我关注进行调节
4. employing a different encoder-decoder architecture 使用不同的编码器-解码器架构

```
Input sequence: [token1, token2, ..., tokenN]
Metadata: [metadata1, metadata2, ..., metadataM]
Self-attention mechanism: [query, key, value]
Output sequence: [output1, output2, ..., outputN]

Condition the self-attention mechanism on the metadata
Inputs: Query Q, Keys K, Values V, Metadata M
Outputs: Context vector C

Compute the self-attention scores: S = QKT
Compute the metadata-aware attention weights: A = MTS
Compute the context vector: C = V*A
    Where:  Q: Query vector
            K: Key vector
            V: Value vector
            M: Metadata vector
            S: Self-attention scores
            A: Metadata-aware attention weights
            C: Context vector
```

一个思路:[MECT: Multi-Metadata Embedding based Cross-Transformer 中文名字识别中加入多个Metadata Embedding](https://aclanthology.org/2021.acl-long.121.pdf)  [代码Code](https://github.com/CoderMusou/MECT4CNER)

2021年  MECT:Multi-metadata Embedding based Cross-Transformer 

为了更好地整合汉字组件的信息，我们使用汉字结构作为另一种元数据，并设计了一种双流形式的多元数据嵌入网络。所建议的网络架构如图2a所示。该方法基于Transformer的编码器结构和FLAT方法，整合了汉语单词的意义和边界信息。所提出的双流模型使用类似于自注意力结构的Cross-Transformer模块来融合汉字组件的信息。在我们的方法中，我们还使用了广泛用于视觉语言任务的多模态协作注意力方法（Lu et al.， 2019）。不同之处在于，我们添加了一个随机初始化的注意力矩阵来计算两种类型的元数据嵌入的注意力偏差。
![img.png](img.png)

word2vec 
emoChord2vec



tokenizer

train

generate



Multi-Metadata Embedding based Cross-Transformer

MECTGan



### Paper [Transformer-based Conditional Variational Autoencoder for Controllable Story Generation](https://arxiv.org/abs/2101.00828)

有VAE的论文介绍，参考


### Paper: [Autoregressive Image Generation using Residual Quantization (CVPR 2022)](https://github.com/kakaobrain/rq-vae-transformer)



### Paper:[TVAE](https://www.ijcai.org/proceedings/2019/0727.pdf)

结构一样，可以借鉴一写描述语句


### Paper: [对话生成TVAE Dialogue Generation](https://arxiv.org/abs/2210.12326)

2022年

![img_1.png](img_1.png)



### Paper:[Generating music with sentiment using Transformer-GANs](https://ar5iv.labs.arxiv.org/html/2212.11134?_immersive_translate_auto_translate=1)  [Code](https://github.com/amazon-science/transformer-gan/blob/main/model/transformer_gan.py)

![img_3.png](img_3.png)
![img_4.png](img_4.png)


