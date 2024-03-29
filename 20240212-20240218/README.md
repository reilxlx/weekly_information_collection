# 01 Sora：OpenAI推出的生成视频的模型
Sora是Diffusion Transformer，核心是Transformer，像训练LLM一样训练视频模型。</br>
把视频压缩成Transformer能处理的patch，对应的就是LLM的token，训练的素材和推理的结果都是patch，最后用Diffusion还原成一帧帧图片。</br>

## Transformer架构</br>
Transformer 是一种深度学习模型，它通过自注意力（self-attention）机制对输入数据间的关系进行建模，这种机制可以让模型更好地处理序列数据。在文本处理领域，transformer 模型能够理解和生成连贯、语义上合理的文本序列。如果应用到视频生成，transformer理论上可以通过类似的方式处理视频帧序列，因为视频可以视作时间序列的图片集合。</br>
## Diffusion模型</br>
Diffusion模型是一类生成模型，其工作原理类似于从噪声数据中逐步生成清晰数据的扩散过程。首先，它们从一个有结构的数据（比如图片）开始，逐步增加噪声，直到数据完全变成随机噪声。生成（或还原）过程则相反，模型逐渐从这个随机噪声中去除噪点，步骤对步骤地恢复出原始数据结构。</br>
在用diffusion模型生成图片的场景中，原理可以这样理解：
学习过程：模型首先学习如何在图片上叠加噪声，将其转化为看似随机的噪声数据。这个过程其实是在学习图像数据和噪声之间的转换关系。</br>
生成过程：随后，生成过程实际上是这个过程的逆过程。从随机噪声数据开始，模型利用它在学习过程中学到的关于噪声和图像数据转换关系的知识，逐步减少噪声并恢复出图像的结构。每一步都会使图像更加清晰，直到最终产出高质量的图片。</br>
如果将这一过程应用于视频，那么视频的每一帧都可以通过diffusion模型从噪声中恢复，而整个视频的连贯性则可以由transformer模型保持。Sora模型这样名为的话，通过transformer捕捉和生成视频帧的时间序列依赖关系，并通过diffusion生成每一帧的细节，从而产生高质量、内容丰富且符合要求的视频。</br>

# 02 BASE TTS：目前最大的文字生成语音模型
BASE TTS 是迄今为止最大的 TTS 模型，在 100K 小时的公共领域语音数据上进行训练，实现了语音自然性的新水平。它部署了一个 10 亿参数的自回归转换器，该转换器将原始文本转换为离散代码（“语音代码”），然后部署一个基于卷积的解码器，该解码器以增量、可流的方式将这些语音代码转换为波形。此外，我们的语音代码是使用一种新颖的语音标记化技术构建的，该技术具有说话人 ID 解缠和字节对编码的压缩功能。与广泛报道的大型语言模型在训练越来越多的数据时的“涌现能力”相呼应，我们表明，使用 10K+ 小时和 500M+ 参数构建的 BASE TTS 变体开始在文本复杂的句子上表现出自然的韵律。我们设计并共享一个专门的数据集来衡量这些文本转语音的紧急能力。我们通过评估包括公开可用的大规模文本转语音系统的基线来展示 BASE TTS 最先进的自然性：YourTTS、Bark 和 TortoiseTTS。</br>
https://www.amazon.science/base-tts-samples/

# 03 DoRA


`````Python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)  # 设置随机种子，以便于结果具有可复制性

# 这个自定义层可以插入到预先训练好的PyTorch模型中，替换现有的nn.Linear层
class DoRALayer(nn.Module):
    def __init__(self, d_in, d_out, rank=4, weight=None, bias=None):
        super().__init__()

        # 根据是否提供权重和偏置进行初始化，或者创建新的参数
        if weight is not None:
            self.weight = nn.Parameter(weight, requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.Tensor(d_out, d_in), requires_grad=False)

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = nn.Parameter(torch.Tensor(d_out), requires_grad=False)

        # 计算权重的列范数
        self.m = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))
        
        # 初始化LoRA矩阵A和B，LoRA是低秩适应的简称
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.lora_A = nn.Parameter(torch.randn(d_out, rank)*std_dev)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_in))

    def forward(self, x):
        # 计算LoRA矩阵的乘积
        lora = torch.matmul(self.lora_A, self.lora_B)
        # 将LoRA调整以获得适应性权重
        adapted = self.weight + lora
        # 对调整后的权重进行标准化
        column_norm = adapted.norm(p=2, dim=0, keepdim=True)
        norm_adapted = adapted / column_norm
        calc_weights = self.m * norm_adapted
        return F.linear(x, calc_weights, self.bias)  # 使用适应性权重进行线性运算


class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, output_dim)  # 简单的线性层

    def forward(self, x):
        x = self.layer1(x)
        return x

# 生成合成数据的函数
def generate_data(num_samples=100, input_dim=10):
    X = torch.randn(num_samples, input_dim)  # 创建随机输入数据
    y = torch.sum(X, dim=1, keepdim=True)  # 创建简单的标签，仅用于演示
    return X, y

# 训练模型的函数
def train(model, criterion, optimizer, data_loader, epochs=5):
    model.train()  # 将模型置于训练模式
    for epoch in range(epochs):  # 迭代训练周期
        for inputs, targets in data_loader:  # 从数据加载器获取数据
            optimizer.zero_grad()  # 清除之前的梯度
            outputs = model(inputs)  # 计算模型输出
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
        #print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def replace_linear_with_dora(model):
    # 递归地遍历模型的所有子层
    for name, module in model.named_children():
        # 如果发现线性层，用DoRALayer替换它
        if isinstance(module, nn.Linear):
            d_in = module.in_features
            d_out = module.out_features
            setattr(model, name, DoRALayer(d_out=d_out, d_in=d_in, weight=module.weight.data.clone(), bias=module.bias.data.clone()))
        else:
            replace_linear_with_dora(module)  # 递归遍历子模块

def print_model_parameters(model):
    # 计算模型的总参数和可训练的参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

# 主脚本
if __name__ == "__main__":
    input_dim, output_dim = 10, 1  # 定义输入输出维度
    model = SimpleModel(input_dim, output_dim)  # 创建模型实例
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = optim.AdamW(model.parameters(), lr=0.001)  # 定义优化器

    X, y = generate_data(num_samples=1000, input_dim=input_dim)  # 生成数据
    dataset = TensorDataset(X, y)  # 创建数据集
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)  # 创建数据加载器

    print_model_parameters(model)  # 打印模型参数

    train(model, criterion, optimizer, data_loader, epochs=100)  # 训练模型

    # 评估模型
    model.eval()  # 将模型置于评估模式
    with torch.no_grad():  # 不追踪梯度
        inputs, targets = next(iter(data_loader))  # 获取一批数据
        predictions = model(inputs)  # 做出预测
        loss = criterion(predictions, targets)  # 计算损失
        print(f"Final Evaluation Loss: {loss.item()}")

    replace_linear_with_dora(model)  # 用DoRALayer替换模型中的Linear层

    print_model_parameters(model)  # 再次打印模型参数

    # 继续训练DoRALayer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    print("Continuing training with DoRA layers...")
    train(model, criterion, optimizer, data_loader, epochs=5)  # 继续训练

    # 再次评估模型
    model.eval()  # 将模型置于评估模式
    with torch.no_grad():  # 不追踪梯度
        inputs, targets = next(iter(data_loader))  # 获取一批数据
        predictions = model(inputs)  # 做出预测
        loss = criterion(predictions, targets)  # 计算损失
        print(f"Final (DoRA) Evaluation Loss: {loss.item()}")  # 打印最终的评估损失
`````

# 04 Magika 毫秒级识别内容类型，识别准确率超99%
Magika 是一种新颖的 AI 驱动的文件类型检测工具，它依靠深度学习的最新进展来提供准确的检测。在引擎盖下，Magika 采用了定制的、高度优化的 Keras 模型，该模型仅重约 1MB，即使在单个 CPU 上运行，也能在几毫秒内实现精确的文件识别。</br>
在超过 1M 个文件和 100 多种内容类型（涵盖二进制和文本文件格式）的评估中，Magika 实现了 99%+ 的精确度和召回率。Magika 用于将 Gmail、云端硬盘和安全浏览文件路由到适当的安全和内容政策扫描程序，从而帮助提高 Google 用户的安全性。</br>
``````python
$ pip install magika
>>> from magika import Magika
>>> m = Magika()
>>> res = m.identify_bytes(b"# Example\nThis is an example of markdown!")
>>> print(res.output.ct_label)
markdown
``````

https://github.com/google/magika

# 05 Open-Source Pre-Processing Tools for Unstructured Data
用于非结构化数据的开源预处理工具</br>
该 unstructured 库提供用于摄取和预处理图像和文本文档（如 PDF、HTML、Word 文档等）的开源组件。unstructured 围绕着简化和优化 的数据处理工作流程LLMs展开。</br>
1. 安装 Python SDK 以支持所有文档类型 pip install "unstructured[all-docs]"
2. 对于不需要任何额外依赖项的纯文本文件、HTML、XML、JSON 和电子邮件，您可以运行 pip install unstructured
3. 要处理其他文档类型，您可以安装这些文档所需的附加功能，例如 pip install "unstructured[docx,pptx]"


libmagic-dev （文件类型检测）</br>
poppler-utils （图片和 PDF）</br>
tesseract-ocr （图像和 PDF，安装 tesseract-lang 以获得额外的语言支持）</br>
libreoffice （MS Office 文档）</br>
pandoc （EPUB、RTF 和 Open Office 文档）</br>

`````python
from unstructured.partition.auto import partition

elements = partition(filename="example-docs/eml/fake-email.eml")
print("\n\n".join([str(el) for el in elements]))
`````
https://github.com/Unstructured-IO/unstructured

# 06 semantic-router 语义路由器
语义路由器是您的LLM和代理的超快决策层。与等待缓慢的LLM生成器进行工具使用决策不同，我们利用语义向量空间的魔力来做出这些决策——使用语义含义来路由我们的请求。
## 安装
`````bash
pip install -qU semantic-router
`````
## 定义一组 Route 对象
`````python
from semantic_router import Route
# we could use this as a guide for our chatbot to avoid political conversations
politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president" "don't you just hate the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)

# this could be used as an indicator to our chatbot to switch to a more
# conversational prompt
chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)

# we place both of our decisions together into single list
routes = [politics, chitchat]
`````

## 初始化一个嵌入/编码器模型
`````python
import os
from semantic_router.encoders import CohereEncoder, OpenAIEncoder

# for Cohere
os.environ["COHERE_API_KEY"] = "<YOUR_API_KEY>"
encoder = CohereEncoder()

# or for OpenAI
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
encoder = OpenAIEncoder()
`````

## 创建一个路由器处理语义决策
`````python
from semantic_router.layer import RouteLayer
rl = RouteLayer(encoder=encoder, routes=routes)
`````

## 使用路由层根据用户查询做出超快决策
`````python
rl("don't you love politics?").name
[Out]: 'politics'
`````
https://github.com/aurelio-labs/semantic-router



# 07 cutword 中文分词库
cutword 是一个中文分词库，字典文件根据截止到2024年1月份的最新数据统计得到，词频更加合理。
分词速度是jieba的两倍。 可通过 python -m cutword.comparewithjieba 进行测试。

Note：本项目并不支持英文实体的识别。如需要英文实体的识别，推荐使用nltk。
`````bash
pip install -U cutword
`````
`````python
from  cutword import Cutter
cutter = Cutter()
res = cutter.cutword("你好，世界")
print(res)
`````
本分词器提供两种词典库，一种是基本的词库，默认加载。一种是升级词库，升级词库总体长度会比基本词库更长一点。

如需要加载升级词库，需要将 want_long_word 设为True
`````python
from  cutword import Cutter

cutter = Cutter()
res = cutter.cutword("精诚所至，金石为开")
print(res) # ['精诚', '所', '至', '，', '金石为开']

cutter = Cutter(want_long_word=True)
res = cutter.cutword("精诚所至，金石为开")
print(res) # ['精诚所至', '，', '金石为开']

`````
