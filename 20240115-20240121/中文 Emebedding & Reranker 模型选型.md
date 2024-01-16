# 中文 Emebedding & Reranker 模型选型
https://github.com/ninehills/blog/issues/111</br>
结论</br>
选型建议：</br>

大部分模型的序列长度是 512 tokens。 8192 可尝试 tao-8k，1024 可尝试 stella。
在专业数据领域上，嵌入模型的表现不如 BM25，但是微调可以大大提升效果。
有微调需求且对模型训练了解较少的，建议选择 bge 系列（完善的训练脚本、负例挖掘等）。但多数模型都基于BERT，训练脚本也通用，其他模型也可以参考。
重排模型选择很少，推荐使用 bge-reranker，也支持微调。reranker 模型因为单次输入较多，只能通过 GPU 部署。

## Embedding 模型
### PEG
作者：腾讯</br>
模型地址： https://huggingface.co/TownsWu/PEG </br>
论文： https://arxiv.org/pdf/2311.11691.pdf</br>
重点优化检索能力。

### GTE 系列
作者：阿里巴巴</br>
模型地址： https://huggingface.co/thenlper/gte-large-zh</br>
论文： https://arxiv.org/abs/2308.03281</br>

### picolo 系列
作者：商汤</br>
地址： https://huggingface.co/sensenova/piccolo-large-zh</br>
有一些微调的小tips</br>

### stella 系列
地址：https://huggingface.co/infgrad/stella-large-zh-v2</br>
博客文章： https://zhuanlan.zhihu.com/p/655322183</br>
基于piccolo 模型fine-tuning，支持1024 序列长度。博客文章记录了一些训练思路。</br>

### BGE 系列
作者：智源研究院</br>
地址：https://huggingface.co/BAAI/bge-large-zh-v1.5</br>
论文：https://arxiv.org/pdf/2309.07597.pdf</br>
Github：https://github.com/FlagOpen/FlagEmbedding</br>
开放信息最多的模型，也提供了fine-tuning 示例代码。同时也是 C-MTEB 榜单的维护者。</br>

### m3e 系列
作者：MokaAI</br>
地址：https://huggingface.co/moka-ai/m3e-large</br>
Github：https://github.com/wangyuxinwhy/uniem</br>
研究的比较早，算是中文通用 Embedding 模型、数据集以及评测比较早的开拓者。</br>

### multilingual-e5-large
地址：https://huggingface.co/intfloat/multilingual-e5-large</br>
论文：https://arxiv.org/pdf/2212.03533.pdf</br>
多语言支持。</br>

### tao-8k
地址： https://huggingface.co/amu/tao-8k</br>
支持8192 序列长度，但是信息很少。</br>

## Reranker 模型
### bge-reranker 系列
作者：智源研究院</br>
地址：https://huggingface.co/BAAI/bge-reranker-large</br>
Github：GitHub - FlagOpen/FlagEmbedding: Dense Retrieval and Retrieval-augmented </br>LLMs

### 基于 xlm-roberta 模型。
alime-reranker-large-zh</br>
地址： https://huggingface.co/Pristinenlp/alime-reranker-large-zh</br>
信息很少。也是基于 xlm-roberta 模型。</br>

C-MTEB</br>
我们只关心 Rerank 和 Retrieval 评测，结果见 mteb</br>