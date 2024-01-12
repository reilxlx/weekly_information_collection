# 01 Comflowy
## 使用介绍
https://www.comflowy.com/zh-CN </br>
https://github.com/BennyKok/comfyui-deploy </br>
https://github.com/comfyanonymous/ComfyUI </br>

在使用stable-diffusion生成图像时，常用table-diffusion-webui
https://github.com/AUTOMATIC1111/stable-diffusion-webui </br>
而 ComfyUI 相比于 Stable Diffusion WebUI 等其他开源产品具备非常强的差异化能力，它具备高度的扩展性和应用可能性，真正做到了让开发者和用户能够根据自己的需求打造个性化的生图流程。

![ComfyUI](./data/comfyui_screenshot.png)

## ComfyUI Lora训练节点
可以直接在Comfy UI中训练Lora模型 </br>
https://www.reddit.com/r/comfyui/comments/193vnxg/lora_training_directly_in_comfyui/?rdt=64578
</br>
</br>
</br>
***
# 02 文本分割的五个层次
我们知道，做 RAG 的时候，文本分块分割相当关键，如何合理的分割文本看似简单，实则细节很多，怎么把相关的信息尽可能保留在一起很重要。
</br>
Greg Kamradt 最近有个视频，详细讲解了文本分割的细节，并且他还整理了一个 Jupyter Notebook，配有代码示例和配图，很是浅显易懂。他把文本分割分成五个层次：</br>
</br>
第 1 层：字符分割 - 对数据进行简单的静态字符划分。</br>
第 2 层：递归字符文本分割 - 基于分隔符列表的递归式分块。</br>
第 3 层：文档特定分割 - 针对不同类型文档（如 PDF、Python、Markdown）的特定分块方法。</br>
第 4 层：语义分割 - 基于嵌入式路径的分块方法。</br>
第 5 层：智能体式分割 - 使用类似智能体的系统来分割文本的实验性方法，适用于你认为 Token 成本接近免费的情况。</br>
https://baoyu.io/translations/rag/5-levels-of-text-splitting </br>
</br>
</br>
</br>
***
# 03 fish-speech
全新的文本转语音(TTS)解决方案，具有高度自定义和灵活性，支持Linux和Windows系统，需要2GB的GPU内存进行推理，使用Flash-Attn进行推理和训练，支持VQGAN和Text2Semantic模型</br>
https://github.com/fishaudio/fish-speech?tab=readme-ov-file </br>