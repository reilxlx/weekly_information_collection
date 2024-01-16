# 01 Surya OCR工具包
Surya 是一个多语言文档 OCR 工具包。它可以做到：（准确的行级文本检测、文本识别、表格和图表检测）</br>
pip install surya-ocr，模型权重将在您第一次运行 surya 时自动下载。</br>
https://github.com/VikParuchuri/surya</br>
![ComfyUI](./data/surya.png)


# 02 DB-GPT-Hub:利用LLMs实现Text-to-SQL
DB-GPT-Hub是一个利用LLMs实现Text-to-SQL解析的实验项目，主要包含数据集收集、数据预处理、模型选择与构建和微调权重等步骤，通过这一系列的处理可以在提高Text-to-SQL能力的同时降低模型训练成本，让更多的开发者参与到Text-to-SQL的准确度提升工作当中，最终实现基于数据库的自动问答能力，让用户可以通过自然语言描述完成复杂数据库的查询操作等工作。

该项目提供数据集、支持多种base模型微调、支持lora和QLora，支持多卡训练，提供不同模型微调的lora_target且包含模型效果评估。</br>
https://github.com/eosphoros-ai/DB-GPT-Hub</br>
![ComfyUI](./data/DBGPT.JPG)

# 03 MarkDownload - Markdown Web Clipper
Chrome插件，用于将网页信息转换为md文件下载</br>
https://chromewebstore.google.com/detail/markdownload-markdown-web/pcmpcfapbekmbjjkdalcgopdkipoggdi</br>

# 04 新手LLM培训指南 --- The Novice's LLM Training Guide.md
[原始版本链接](https://rentry.org/llm-training#the-basics)https://rentry.org/llm-training#the-basics</br>
[双语翻译版本,新手LLM培训指南 --- The Novice's LLM Training Guide](./新手LLM培训指南%20---%20The%20Novice's%20LLM%20Training%20Guide.md)</br>
![llm](./data/llm.JPG)

# 05 Infinity 
Infinity 是一个高吞吐量、低延迟的 REST API，用于提供向量嵌入，支持各种句子转换器模型和框架。</br>
pip install infinity-emb[all]</br>
```python
import asyncio
from infinity_emb import AsyncEmbeddingEngine

sentences = ["Embed this is sentence via Infinity.", "Paris is in France."]
engine = AsyncEmbeddingEngine(model_name_or_path = "BAAI/bge-small-en-v1.5", engine="torch")

async def main(): 
    async with engine: # engine starts with engine.astart()
        embeddings, usage = await engine.embed(sentences=sentences)
    # engine stops with engine.astop()
asyncio.run(main())
```

https://github.com/michaelfeil/infinityhttps://github.com/michaelfeil/infinity</br>

# 06 Frontend-only live semantic search with transformers.js
直接在浏览器中进行语义搜索！使用 transformers.js 和 sentence-transformers/all-MiniLM-L6-v2 的量化版本，在没有服务器端推理的情况下计算嵌入和余弦相似度。</br>
数据隐私友好 - 您输入的文本数据不会发送到服务器，而是保留在您的浏览器中！</br>
https://github.com/do-me/SemanticFinder</br>