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


# 07 bce-reranker-base_v1 + bce-embedding-base_v1
BCEmbedding是由网易有道开发的双语和跨语种语义表征算法模型库，其中包含EmbeddingModel和RerankerModel两类基础模型。EmbeddingModel专门用于生成语义向量，在语义搜索和问答中起着关键作用，而RerankerModel擅长优化语义搜索结果和语义相关顺序精排。</br>

bce-embedding-base_v1 主要特点(Key Features)：
中英双语，以及中英跨语种能力(Bilingual and Crosslingual capability in English and Chinese)；
RAG优化，适配更多真实业务场景(RAG adaptation for more domains)；
方便集成进langchain和llamaindex(Easy integrations for langchain and llamaindex)。</br>

bce-reranker-base_v1 主要特点(Key Features)：
中英日韩四个语种，以及中英日韩四个语种的跨语种能力(Multilingual and Crosslingual capability in English, Chinese, Japanese and Korean)；
RAG优化，适配更多真实业务场景(RAG adaptation for more domains)；
适配长文本做rerank(Fix the reranking strategy for long passages)。</br>
https://huggingface.co/maidalun1020/bce-reranker-base_v1</br>
https://huggingface.co/maidalun1020/bce-embedding-base_v1</br>

# 08 中文 Emebedding & Reranker 模型选型
[中文 Emebedding & Reranker 模型选型](./中文%20Emebedding%20&%20Reranker%20模型选型.md)

# 09 Transform Screenshots into HTML Code
UI界面截图转换成HTML代码。
包含 823,000 对网站屏幕截图和 HTML/CSS 代码的数据集。 
Websight 旨在训练视觉语言模型 （VLM） 以将图像转换为代码。</br>
数据集：https://huggingface.co/datasets/HuggingFaceM4/WebSight</br>
模型：  https://huggingface.co/HuggingFaceM4/VLM_WebSight_finetuned</br>

# 10 音频降噪增强模型
Resemble Enhance 是一种 AI 驱动的工具，旨在通过执行降噪和增强来提高整体语音质量。它由两个模块组成：一个降噪器，用于将语音与嘈杂的音频分离，以及一个增强器，通过恢复音频失真和扩展音频带宽来进一步提高感知音频质量。这两个模型在高质量的 44.1kHz 语音数据上进行训练，可保证以高质量增强您的语音。

安装稳定版本：
 ```
pip install resemble-enhance --upgrade
 ```
增强：
 ```
resemble_enhance in_dir out_dir
 ```
降噪：
 ```
resemble_enhance in_dir out_dir --denoise_only
 ```
 https://github.com/resemble-ai/resemble-enhance

 # 11 whisper针对大型语音数据集再次训练
 数据集：speech_asr/speech_asr_aishell1_trainsets</br>
 数据集简介
用于“Aishell-1学术数据集的中文语音识别模型”的Aishell-1训练集。

希尔贝壳中文普通话开源语音数据库AISHELL-ASR0009-OS1录音时长178小时，是希尔贝壳中文普通话语音数据库AISHELL-ASR0009的一部分。AISHELL-ASR0009录音文本涉及智能家居、无人驾驶、工业生产等11个领域。录制过程在安静室内环境中， 同时使用3种不同设备： 高保真麦克风（44.1kHz，16-bit）；Android系统手机（16kHz，16-bit）；iOS系统手机（16kHz，16-bit）。高保真麦克风录制的音频降采样为16kHz，用于制作AISHELL-ASR0009-OS1。400名来自中国不同口音区域的发言人参与录制。经过专业语音校对人员转写标注，并通过严格质量检验，此数据库文本正确率在95%以上。分为训练集、开发集、测试集。（支持学术研究，未经允许禁止商用。）

# 12 opencompass
系统性评估大模型在语言、知识、推理、创作、长文本和智能体等多个能力维度的表现。</br>
安装
 ```
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
 ```
使用，下载数据集到 opencompass/data/ 处</br>
 ```
wget https://github.com/open-compass/opencompass/releases/download/0.1.8.rc1/OpenCompassData-core-20231110.zip</br>
unzip OpenCompassData-core-20231110.zip
 ```
OpenCompass 预定义了许多模型和数据集的配置，你可以通过 工具 列出所有可用的模型和数据集配置。</br>
 ```
 # 列出所有配置
python tools/list_configs.py
# 列出所有跟 llama 及 mmlu 相关的配置
python tools/list_configs.py llama mmlu
 ```
 测评模型</br>
  ```
 python run.py --models hf_llama_7b --datasets mmlu_ppl ceval_ppl
  ```


https://github.com/open-compass/opencompass/blob/main/README_zh-CN.md

# 13 AWPortrait civitai模型
![AWPortrait](./data/civitai-model-Awportrait1.3.JPG)
https://civitai.com/models/61170?modelVersionId=304593</br>


# 14 中文Chinese-Mixtral-8x7B
本项目基于Mistral发布的模型Mixtral-8x7B进行了中文扩词表增量预训练，希望进一步促进中文自然语言处理社区对MoE模型的研究。我们扩充后的词表显著提高了模型对中文的编解码效率，并通过大规模开源语料对扩词表模型进行增量预训练，使模型具备了强大的中文生成和理解能力。

项目开源内容：</br>
1、中文Mixtral-8x7B扩词表大模型</br>
2、扩词表增量预训练代码</br>
https://github.com/HIT-SCIR/Chinese-Mixtral-8x7B

分别使用以下评测数据集对Chinese-Mixtral-8x7B进行评测：</br>
C-Eval：一个全面的中文基础模型评估套件。它包含了13948个多项选择题，涵盖了52个不同的学科和四个难度级别。</br>
CMMLU：一个综合性的中文评估基准，专门用于评估语言模型在中文语境下的知识和推理能力，涵盖了从基础学科到高级专业水平的67个主题。</br>
MMLU：一个包含57个多选任务的英文评测数据集，涵盖了初等数学、美国历史、计算机科学、法律等，难度覆盖高中水平到专家水平，是目前主流的LLM评测数据集之一。</br>
HellaSwag：一个极具挑战的英文NLI评测数据集，每一个问题都需要对上下文进行深入理解，而不能基于常识进行回答。</br>

合并QLora模型</br>
ChrisHayduk/merge_qlora_with_quantized_model.py</br>
https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930</br>

微调模型</br>
https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py</br>
https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/blob/main/scripts/training/run_pt.sh

