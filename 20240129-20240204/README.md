# 01 500行代码构建搜索引擎
- Built-in support for LLM
- Built-in support for search engine
- Customizable pretty UI interface
- Shareable, cached search results

https://github.com/leptonai/search_with_lepton

# 02 RSS-to-Telegram-Bot关心你的阅读体验的 Telegram RSS 机器人
https://github.com/Rongronggg9/RSS-to-Telegram-Bot?tab=readme-ov-file

# 03 makeMoE：从头开始实现专家语言模型的稀疏混合
这篇博客介绍了如何从头开始实现专家语言模型的稀疏混合。这主要受到 Andrej Karpathy 项目“makemore”的启发，并在很大程度上基于该项目，并从该实现中借用了许多可重用的组件。就像makemore一样，makeMoE也是一个自回归的字符级语言模型，但使用了前面提到的稀疏的专家架构。博客的其余部分重点介绍此体系结构的关键元素以及它们的实现方式。我的目标是让你在阅读此博客并逐步完成存储库中的代码后，直观地了解它是如何工作的。
https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch

# 04 Code Llama
Code Llama 是 Llama 2 的代码专用版本，它是通过在其特定于代码的数据集上进一步训练 Llama 2 而创建的，从同一数据集中采样更多数据的时间更长。从本质上讲，Code Llama 具有增强的编码功能。它可以从代码和自然语言提示中生成代码和有关代码的自然语言（例如，“给我写一个输出斐波那契数列的函数”）。它还可用于代码完成和调试。它支持当今使用的许多最流行的编程语言，包括 Python、C++、Java、PHP、Typescript （Javascript）、C#、Bash 等。

有三种大小（7b、13b、34b）以及三种风格（基本型号、Python 微调和指令调整）。
https://huggingface.co/codellama

# 05 30多种矢量数据库功能、性能和价格对比
https://vdbs.superlinked.com/?utm_source=twitterads&utm_medium=cpc&utm_campaign=twitter_keywords_v1&utm_id=twitter_keywords_v1&utm_term=vectordb+embeddings&twclid=2-5mtt3rwyxp1g8ls65ovctjboz

# 06 nomic-embed-text-v1: A Reproducible Long Context (8192)  Text Embedder
nomic-embed-text-v1 是 8192 上下文长度文本编码器，在短上下文任务和长上下文任务上的表现优于 OpenAI text-embedding-ada-002 和 text-embedding-3-small。.

https://huggingface.co/nomic-ai/nomic-embed-text-v1