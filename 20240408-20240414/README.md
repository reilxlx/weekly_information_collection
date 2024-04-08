# 01 twitter-web-exporter
"twitter-web-exporter" 是一个GitHub项目，它允许用户从Twitter网页应用直接导出推文、书签、列表等数据。用户可以在保证数据隐私的情况下，将数据导出为JSON、CSV、HTML等格式，包括推文、喜欢、关注者列表、以及搜索结果等，而无需开发者账号或API密钥。该工具免费、开源，通过浏览器扩展（如Tampermonkey或Violentmonkey）进行安装使用。
该项目的抓取需要浏览页面，如果页面数据未加载则无法抓取后续数据。

https://github.com/prinsss/twitter-web-exporter

# 02 Twitter-Insight-LLM
项目"Twitter-Insight-LLM"旨在通过Selenium抓取Twitter上的喜欢的推文，并将其保存为JSON和Excel文件，之后进行初步数据分析和图像标题生成。这是利用大型语言模型（LLM）进行更大个人项目的初步步骤。技术包括Python、Selenium和OpenAI API，用于数据抓取、保存、分析以及生成图像标题。项目支持将数据分析结果可视化，并允许通过OpenAI API为推文图像生成标题。

https://github.com/AlexZhangji/Twitter-Insight-LLM?tab=readme-ov-file

# 03 video-subtitle-remover
"video-subtitle-remover"是一个基于AI的工具，用于从图片或视频中去除硬编码字幕和类似文本的水印。它能够无损去除字幕和水印，同时保持原有分辨率，并支持自定义字幕位置去除或自动去除全视频的所有文本。该项目不需要第三方API，可以在本地运行，使用Python编写，并依赖于CUDA和cuDNN进行GPU加速。

https://github.com/YaoFANGUK/video-subtitle-remover

# 04 cutword中文分词工具
"cutword"是一个高效的中文分词和命名实体识别工具，旨在提供比jieba库更快的分词速度和更新的字典文件。它支持自定义字典，提供基础和升级词库选项，并允许用户选择是否返回分词结果。此外，"cutword"提供命名实体识别功能，识别多种类型的实体，如食品、组织、事件等，但不支持英文实体的识别。如需要英文实体的识别，推荐使用nltk（https://github.com/nltk/nltk）。

https://github.com/liwenju0/cutword