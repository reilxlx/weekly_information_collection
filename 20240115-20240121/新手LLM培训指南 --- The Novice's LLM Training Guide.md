[![lmg-train.png](https://s8d3.turboimg.net/sp/e6b7e8c2bb66fa33a62439ac77134740/lmg-train.png "lmg-train.png")](https://www.turboimagehost.com/p/89494772/lmg-train.png.html)

**Written by [Alpin](https://github.com/AlpinDale) 作者通過： Alpin** **Inspired by [/hdg/'s LoRA train rentry](https://rentry.org/lora_train)  
灵感来自 /hdg/ 的 LoRA 火车 rentry**

This guide is being slowly updated. We've already moved to the axolotl trainer.  
本指南正在缓慢更新。我们已经转向了蝾螈训练器。

___

1.  [The Basics 基础知识](https://rentry.org/llm-training#the-basics)
    1.  [The Transformer architecture  
        Transformer 架构](https://rentry.org/llm-training#the-transformer-architecture)
2.  [Training Basics 培训基础](https://rentry.org/llm-training#training-basics)
    1.  [Pre-training 预训练](https://rentry.org/llm-training#pre-training)
    2.  [Fine-tuning 微调](https://rentry.org/llm-training#fine-tuning)
    3.  [Low-Rank Adaptation (LoRA)  
        低秩适应 （LoRA）](https://rentry.org/llm-training#low-rank-adaptation-lora)
3.  [Fine-tuning 微调](https://rentry.org/llm-training#fine-tuning_1)
    1.  [Training Compute 训练计算](https://rentry.org/llm-training#training-compute)
    2.  [Gathering a Dataset 收集数据集](https://rentry.org/llm-training#gathering-a-dataset)
    3.  [Dataset structure 数据集结构](https://rentry.org/llm-training#dataset-structure)
    4.  [Processing the raw dataset  
        处理原始数据集](https://rentry.org/llm-training#processing-the-raw-dataset)
        1.  [HTML \[HTML全文\]](https://rentry.org/llm-training#html)
        2.  [CSV](https://rentry.org/llm-training#csv)
        3.  [SQL](https://rentry.org/llm-training#sql)
    5.  [Minimizing the noise 将噪音降至最低](https://rentry.org/llm-training#minimizing-the-noise)
    6.  [Starting the training run  
        开始训练运行](https://rentry.org/llm-training#starting-the-training-run)
4.  [Low-Rank Adaptation (LoRA)  
    低秩适应 （LoRA）](https://rentry.org/llm-training#low-rank-adaptation-lora_1)
    1.  [LoRA hyperparameters LoRA 超参数](https://rentry.org/llm-training#lora-hyperparameters)
        1.  [LoRA Rank LoRA排名](https://rentry.org/llm-training#lora-rank)
        2.  [LoRA Alpha LoRA 阿尔法](https://rentry.org/llm-training#lora-alpha)
        3.  [LoRA Target Modules LoRA 目标模块](https://rentry.org/llm-training#lora-target-modules)
5.  [QLoRA](https://rentry.org/llm-training#qlora)
6.  [Training Hyperparameters  
    训练超参数](https://rentry.org/llm-training#training-hyperparameters)
    1.  [Batch Size and Epoch  
        批量大小和纪元](https://rentry.org/llm-training#batch-size-and-epoch)
        1.  [Stochastic Gradient Descent  
            随机梯度下降](https://rentry.org/llm-training#stochastic-gradient-descent)
        2.  [Sample 样本](https://rentry.org/llm-training#sample)
        3.  [Batch 批](https://rentry.org/llm-training#batch)
        4.  [Epoch 时代](https://rentry.org/llm-training#epoch)
        5.  [Batch vs Epoch 批处理与纪元](https://rentry.org/llm-training#batch-vs-epoch)
    2.  [Learning Rate 学习率](https://rentry.org/llm-training#learning-rate)
        1.  [Learning Rate and Gradient Descent  
            学习率和梯度下降](https://rentry.org/llm-training#learning-rate-and-gradient-descent)
        2.  [Configuring the Learning Rate  
            配置学习率](https://rentry.org/llm-training#configuring-the-learning-rate)
    3.  [Gradient Accumulation 梯度累积](https://rentry.org/llm-training#gradient-accumulation)
        1.  [Backpropagation 反向传播](https://rentry.org/llm-training#backpropagation)
        2.  [Gradient Accumulation explained  
            梯度累积解释](https://rentry.org/llm-training#gradient-accumulation-explained)
        3.  [Iteration 迭 代](https://rentry.org/llm-training#iteration)
        4.  [Configuring the number of gradient accumulation steps  
            配置梯度累积步骤数](https://rentry.org/llm-training#configuring-the-number-of-gradient-accumulation-steps)
7.  [Interpreting the Learning Curves  
    解释学习曲线](https://rentry.org/llm-training#interpreting-the-learning-curves)
    1.  [Overview 概述](https://rentry.org/llm-training#overview)
    2.  [Model Behaviour Diagnostics  
        模型行为诊断](https://rentry.org/llm-training#model-behaviour-diagnostics)
        1.  [Underfit Learning Curves  
            欠拟合学习曲线](https://rentry.org/llm-training#underfit-learning-curves)
        2.  [Overfit Learning Curves 过拟合学习曲线](https://rentry.org/llm-training#overfit-learning-curves)
        3.  [Well-fit Learning Curves  
            拟合良好的学习曲线](https://rentry.org/llm-training#well-fit-learning-curves)

___

## The Basics 基础知识

The most common architecture used for language modeling is the Transformer architecture, introduced by Vaswani et al. in the famous paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). We won't go into the specifics of the architecture here, as we'd have to discuss all the older technologies that led to and contributed to its creation. Transformers allow us to train Large Language Models (LLMs) with incredible reasoning capabilities, while keeping the architecture simple enough for a novice in machine learning to get started on training/playing around with them.  
用于语言建模的最常见架构是 Transformer 架构，由 Vaswani 等人在著名论文 Attention Is All You Need 中介绍。我们不会在这里详细介绍架构的细节，因为我们必须讨论导致并促成其创建的所有旧技术。Transformer 允许我们训练具有令人难以置信的推理能力的大型语言模型 （LLMs），同时保持架构足够简单，以便机器学习新手可以开始训练/玩弄它们。

The most common language used for training and building Transformer models is Python, a very high-level (i.e. further away from raw machine code) language. This makes it easy for the layman to get familiar with the process. The most popular library in use is the [HuggingFace Transformers](https://github.com/huggingface/transformers), which serves as the backbone of almost every LLM trainer today.  
用于训练和构建 Transformer 模型的最常用语言是 Python，这是一种非常高级（即远离原始机器代码）的语言。这使外行人很容易熟悉该过程。最流行的库是 HuggingFace 变形金刚，它是当今几乎所有LLM训练师的支柱。

LLMs are, in essence, a lossy form of text compression. We create tensors (multi-dimensional matrices) with random values and parameters, then feed them an exorbitant amount of text data (in terabytes!) so they can learn the relationship between all the data and recognize patterns between them. All these patterns are stored in the tensors we've randomly initialized as probabilities - the model learns how probable it is for a specific word to be followed by another one, and so on. A very high-level definition of LLMs would be "compressing the probability distribution of a language, such as English, into a collection of matrices."  
LLMs从本质上讲，是一种有损形式的文本压缩。我们创建具有随机值和参数的张量（多维矩阵），然后向它们提供大量文本数据（以 TB 为单位！），以便它们可以学习所有数据之间的关系并识别它们之间的模式。所有这些模式都存储在我们随机初始化为概率的张量中——模型学习一个特定单词后面跟着另一个单词的可能性，依此类推。一个非常高级的定义LLMs是“将一种语言（如英语）的概率分布压缩到矩阵的集合中”。

For example, if you input: "How are" into the LLM, it'll calculate how probable the next word is. For example, it may assign a probability of 60% to "you?", 20% "things", and so on.  
例如，如果您在 中输入：“How are”LLM，它将计算下一个单词的可能性。例如，它可能为“你”分配 60% 的概率，为“事物”分配 20% 的概率，依此类推。

The random initialization discussed above is largely not applicable to us, as it's _very_ expensive (we're talking about millions of dollars for the larger models). This rentry will go over _fine-tuning_ the models - that is, taking a pre-trained model and feeding it a small amount of data, usually a few MBs, to align its behaviour towards whatever downstream task you have in mind. For example, if you want a coding assistant model, you would fine-tune the model on coding examples, and so on.  
上面讨论的随机初始化在很大程度上不适用于我们，因为它非常昂贵（我们谈论的是大型模型的数百万美元）。这个 rentry 将对模型进行微调 - 也就是说，采用预先训练的模型并为其提供少量数据（通常为几 MB），以使其行为与您想到的任何下游任务保持一致。例如，如果需要编码助手模型，则可以根据编码示例对模型进行微调，依此类推。

### The Transformer architecture  
Transformer 架构[](https://rentry.org/llm-training#the-transformer-architecture "Permanent link")

It's always good practice to have an understanding of what you're working with, though it's not _strictly_ necessary for fine-tuning purposes, since you'll be running scripts that call the Transformers library's `Trainer` class.  
了解您正在使用的内容始终是一种很好的做法，尽管对于微调目的来说，这并不是绝对必要的，因为您将运行调用 Transformers 库类的 `Trainer` 脚本。

The best source is, of course, the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper. It introduced the Transformer architecture and is a profoundly important paper to read through. You might, however, need to read these first, since the authors assume you already have a basic grasp of neural networks. I recommend reading these in order:  
当然，最好的来源是“注意力就是你所需要的一切”纸。它介绍了 Transformer 架构，是一篇非常重要的论文。但是，您可能需要先阅读这些内容，因为作者假设您已经对神经网络有了基本的了解。我建议按顺序阅读这些内容：

-   [Deep Learning in Neural Networks: an Overview  
    神经网络中的深度学习：概述](https://arxiv.org/abs/1404.7828)
-   [An Introduction to Convolutional Neural Networks  
    卷积神经网络简介](https://arxiv.org/abs/1511.08458)
-   [Recurrent Neural Networks (RNNs): A gentle Introduction and Overview  
    递归神经网络 （RNN）：温和的介绍和概述](https://arxiv.org/abs/1912.05911)

Paper too hard to read?  
纸张太难阅读？

You're not alone. Academics tend to intentionally obfuscate their papers. You can always look for blog posts or articles on each topic, where they tend to provide easy to digest explanations. One great resource is HuggingFace blogposts.  
你并不孤单。学者们倾向于故意混淆他们的论文。您可以随时查找有关每个主题的博客文章或文章，它们往往会提供易于理解的解释。一个很好的资源是 HuggingFace 博客文章。

___

## Training Basics 培训基础[](https://rentry.org/llm-training#training-basics "Permanent link")

There's essentially three (3) approaches to training LLMs: pre-training, fine-tuning, and LoRA/Q-LoRA.  
基本上有三 （3） 种训练方法：预训练LLMs、微调和 LoRA/Q-LoRA。

### Pre-training 预训练[](https://rentry.org/llm-training#pre-training "Permanent link")

Pre-training involves several steps. First, a massive dataset of text data, often in terabytes, is gathered. Next, a model architecture is chosen or created specifically for the task at hand. Additionally, a tokenizer is trained to appropriately handle the data, ensuring that it can efficiently encode and decode text. The dataset is then pre-processed using the tokenizer's vocabulary, converting the raw text into a format suitable for training the model. This step involves mapping tokens to their corresponding IDs, and incorporating any necessary special tokens or attention masks. Once the dataset is pre-processed, it is ready to be used for the pre-training phase.  
预训练包括几个步骤。首先，收集大量文本数据，通常以 TB 为单位。接下来，专门为手头的任务选择或创建模型架构。此外，还训练分词器适当地处理数据，确保它能够有效地对文本进行编码和解码。然后，使用分词器的词汇表对数据集进行预处理，将原始文本转换为适合训练模型的格式。此步骤涉及将令牌映射到其相应的 ID，并合并任何必要的特殊令牌或注意力掩码。对数据集进行预处理后，即可将其用于预训练阶段。

During pre-training, the model learns to predict the next word in a sentence or to fill in missing words by utilizing the vast amount of data available. This process involves optimizing the model's parameters through an iterative training procedure that maximizes the likelihood of generating the correct word or sequence of words given the context.  
在预训练期间，该模型学习预测句子中的下一个单词或利用大量可用数据来填补缺失的单词。此过程涉及通过迭代训练过程优化模型的参数，该过程在给定上下文中最大限度地提高生成正确单词或单词序列的可能性。

To accomplish this, the pre-training phase typically employs a variant of the self-supervised learning technique. The model is presented with partially masked input sequences, where certain tokens are intentionally hidden, and it must predict those missing tokens based on the surrounding context. By training on massive amounts of data in this manner, the model gradually develops a rich understanding of language patterns, grammar, and semantic relationships. This specific approach is for [Masked Language Modeling](https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling). The most commonly used method today, however, is [Causal Language Modeling](https://huggingface.co/docs/transformers/main/tasks/language_modeling). Unlike masked language modeling, where certain tokens are masked and the model predicts those missing tokens, causal language modeling focuses on predicting the next word in a sentence given the preceding context.  
为了实现这一点，预训练阶段通常采用自监督学习技术的变体。该模型呈现部分屏蔽的输入序列，其中某些标记被故意隐藏，并且它必须根据周围的上下文预测这些缺失的标记。通过以这种方式对大量数据进行训练，该模型逐渐对语言模式、语法和语义关系有了丰富的理解。此特定方法适用于掩码语言建模。然而，当今最常用的方法是因果语言建模。与掩码语言建模不同，在掩码语言建模中，某些标记被掩蔽，模型预测那些缺失的标记，而因果语言建模侧重于在给定前面的上下文的情况下预测句子中的下一个单词。

This initial pre-training phase aims to capture general language knowledge, making the model a proficient language encoder. But, unsurprisingly, it lacks specific knowledge about a particular task or domain. To bridge this gap, a subsequent fine-tuning phase follows pre-training.  
这个初始的预训练阶段旨在捕获一般语言知识，使模型成为熟练的语言编码器。但是，不出所料，它缺乏有关特定任务或领域的特定知识。为了弥合这一差距，在预训练之后进行后续的微调阶段。

### Fine-tuning 微调[](https://rentry.org/llm-training#fine-tuning "Permanent link")

After the initial pre-training phase, where the model learns general language knowledge, fine-tuning allows us to specialize the model's capabilities and optimize its performance on a narrower, task-specific dataset.  
在最初的预训练阶段之后，模型学习一般语言知识，微调使我们能够专门化模型的功能，并在更窄的、特定于任务的数据集上优化其性能。

The process of fine-tuning involves several key steps. Firstly, a task-specific dataset is gathered, consisting of labeled examples relevant to the desired task. For example, if the task is instruct-tuning, a dataset of instruction-response pair is gathered. The fine-tuning dataset size is significantly smaller than the sets typically used for pre-training.  
微调过程涉及几个关键步骤。首先，收集特定于任务的数据集，该数据集由与所需任务相关的标记示例组成。例如，如果任务是指令调整，则收集指令-响应对的数据集。微调数据集的大小明显小于通常用于预训练的数据集。

Next, the pre-trained model is initialized with its previously learned parameters. The model is then trained on the task-specific dataset, optimizing its parameters to minimize a task-specific loss function (i.e. how "off" the model was from the desired result).  
接下来，使用先前学习的参数初始化预训练模型。然后，在特定于任务的数据集上训练模型，优化其参数以最小化特定于任务的损失函数（即模型与预期结果的“偏离”程度）。

During fine-tuning, the parameters of the pre-trained model are adjusted using gradient-based optimization algorithms such as [stochastic gradient descent (SGD)](https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31) or [Adam](https://arxiv.org/abs/1412.6980). The gradients are computed by backpropagating the loss through the model's layers, allowing the model to learn from its mistakes and update its parameters accordingly.  
在微调过程中，使用基于梯度的优化算法（如随机梯度下降 （SGD） 或 Adam）调整预训练模型的参数。梯度是通过将损失反向传播到模型的层来计算的，使模型能够从错误中吸取教训并相应地更新其参数。

To enhance the fine-tuning process, additional techniques can be employed, such as [learning rate scheduling](https://d2l.ai/chapter_optimization/lr-scheduler.html), regularization methods like [dropout](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/) or [weight decay](https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab), or [early stopping](https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/) to prevent [overfitting](https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/). These techniques help optimize the model's generalization and prevent it from memorizing the training dataset too closely.  
为了增强微调过程，可以采用其他技术，例如学习速率调度、正则化方法（如辍学或权重衰减）或提前停止以防止过度拟合。这些技术有助于优化模型的泛化，并防止其过于紧密地记忆训练数据集。

### Low-Rank Adaptation (LoRA)  
低秩适应 （LoRA）[](https://rentry.org/llm-training#low-rank-adaptation-lora "Permanent link")

Fine-tuning is computationally expensive, requiring hundreds of GBs of VRAM for training multi-billion parameter models. To solve this specific problem, a new method was proposed: Low-Rank Adaptation. Compared to fine-tuning OPT-175B with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirements by over 3 times. Refer to the paper [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) and the HuggingFace [PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models on Low-Resource Hardware](https://huggingface.co/blog/peft) blog post.  
微调的计算成本很高，需要数百 GB 的 VRAM 来训练数十亿个参数模型。为了解决这一具体问题，提出了一种新的方法：低秩自适应。与使用 Adam 微调 OPT-175B 相比，LoRA 可以将可训练参数的数量减少 10,000 倍，将 GPU 内存需求减少 3 倍以上。请参阅论文 LoRA： Low-Rank Adaptation of Large Language Models and the HuggingFace PEFT： Parameter-Efficient Fine-Tuning of Billion-Scale Models on Low-Resource Hardware 博客文章。

A 3x memory requirement reduction is still in the realm of unfeasible for the average consumer. Fortunately, a new LoRA training method was introduced: Quantized Low-Rank Adaptation (QLoRA). It leverages the [bitsandbytes](https://github.com/timdettmers/bitsandbytes) library for on-the-fly and near-lossless quantization of language models and applies it to the LoRA training procedure. This results in **massive** reductions in memory requirement - enabling the training of models as large as 70 billion parameters on 2x NVIDIA RTX 3090s! For comparison, you would normally require over 16x A100-80GB GPUs for fine-tuning a model of that size class; the associated cost would be tremendous.  
对于普通消费者来说，将内存需求降低 3 倍仍然不可行。幸运的是，引入了一种新的 LoRA 训练方法：量化低秩适应 （QLoRA）。它利用 bitsandbytes 库对语言模型进行动态和近乎无损的量化，并将其应用于 LoRA 训练程序。这大大降低了内存需求 - 能够在 2 个 NVIDIA RTX 3090 上训练多达 700 亿个参数的模型！相比之下，您通常需要超过 16 个 A100-80GB GPU 来微调该尺寸级别的模型;相关的成本将是巨大的。

This next sections of this rentry will focus on the fine-tuning and LoRA/QLoRA methods.  
本教程的下一节将重点介绍微调和 LoRA/QLoRA 方法。

___

## Fine-tuning 微调[](https://rentry.org/llm-training#fine-tuning_1 "Permanent link")

As explained earlier, fine-tuning can be expensive, depending on the model size you choose. You typically want at least 6B/7B parameters. We'll go through some options for acquiring training compute.  
如前所述，微调可能很昂贵，具体取决于您选择的模型大小。通常至少需要 6B/7B 参数。我们将介绍一些获取训练计算的选项。

### Training Compute 训练计算[](https://rentry.org/llm-training#training-compute "Permanent link")

Depending on your model and dataset size, the memory requirement will vary. You can refer to EleutherAI's [Transformer Math 101](https://blog.eleuther.ai/transformer-math) blog post for detailed, but easy to understand, calculations.  
根据模型和数据集大小，内存要求会有所不同。您可以参考 EleutherAI 的 Transformer Math 101 博客文章，了解详细但易于理解的计算。

You will want to fine-tune a model of _at least_ the 7B class. Some popular options are [Llama-2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) and [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1), etc. This size class typically requires memory in the 160~192GB range. Your options, essentially, boil down to:  
您需要微调至少为 7B 类的模型。一些流行的选项是 Llama-2 7B 和 Mistral 7B 等。此大小级别通常需要 160~192GB 范围内的内存。从本质上讲，您的选择可以归结为：

-   Renting GPUs from cloud services; e.g. [Runpod](https://runpod.io/), [VastAI](https://vast.ai/), [Lambdalabs](https://lambdalabs.com/), and [Amazon AWS Sagemaker](https://aws.amazon.com/sagemaker/).  
    从云服务租用 GPU;例如 Runpod、VastAI、Lambdalabs 和 Amazon AWS Sagemaker。  
    Out of the four examples, VastAI is the cheapest (but also the least reliable), while Amazon Sagemaker is the most expensive. I recommend using either Runpod or Lambdalabs.  
    在四个示例中，VastAI 是最便宜的（但也是最不可靠的），而 Amazon Sagemaker 是最昂贵的。我建议使用 Runpod 或 Lambdalabs。
-   Using Google's TPU Research Cloud  
    使用 Google 的 TPU Research Cloud  
    You can apply for **free** access to the Google TRC program and potentially receive up to 110 TPU machines. Keep in mind that TPUs are significantly different, architecture-wise, from GPUs; you will need to learn how they work first before committing to your 30-day free TRC plan. Fortunately, google provides free access to TPUs (albeit weak ones) via [Google Colaboratory](https://colab.research.google.com/). There are also libraries and guides made for fine-tuning LLMs on TPUs. The [Mesh Transformers JAX](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/howto_finetune.md) repository has a guide for fine-tuning GPT-J models on TRC, and [EasyLM](https://github.com/young-geng/EasyLM) provides an _easy_ way to pre-train, fine-tune and evaluate language models on both TPUs and GPUs.  
    您可以申请免费访问 Google TRC 计划，并可能获得多达 110 台 TPU 机器。请记住，TPU 在架构方面与 GPU 有很大不同;在承诺使用 30 天免费 TRC 计划之前，您需要先了解它们的工作原理。幸运的是，谷歌通过 Google Colaboratory 提供免费的 TPU（尽管很弱）。此外，还有用于对 TPU 进行微调LLMs的库和指南。Mesh Transformers JAX 存储库提供了在 TRC 上微调 GPT-J 模型的指南，而 EasyLM 提供了一种在 TPU 和 GPU 上预训练、微调和评估语言模型的简单方法。
-   Know a guy who knows a guy.  
    认识一个认识男人的人。  
    Maybe one of your friends knows someone with access to a supercomputer. Wouldn't that be cool?  
    也许你的一个朋友认识一个可以使用超级计算机的人。那不是很酷吗？

### Gathering a Dataset 收集数据集[](https://rentry.org/llm-training#gathering-a-dataset "Permanent link")

Dataset gathering is, without a doubt, the most important part of your fine-tuning journey. Both quality and quantity matter - though quality is more important.  
毫无疑问，数据集收集是微调过程中最重要的部分。质量和数量都很重要——尽管质量更重要。

First, think about what you want the fine-tuned model to do. Write stories? Role-play? Write your emails for you? Maybe you want to create your AI waifubot. For the purposes of this rentry, let's assume you want to train a chat & roleplaying model, like [Pygmalion](https://huggingface.co/PygmalionAI). You'll want to gather a dataset of **conversations.** Specifically, conversations in the style of internet RP. The gathering part might be quite challenging; you'll have to figure that out yourself :D  
首先，想想你希望微调后的模型做什么。 写故事？角色扮演？为你写你的电子邮件？也许你想创建你的 AI waifubot。为了这个 rentry，让我们假设你想训练一个聊天和角色扮演模型，比如 Pygmalion。您需要收集对话数据集。具体来说，是互联网 RP 风格的对话。聚会部分可能非常具有挑战性;你必须自己弄清楚:D

### Dataset structure 数据集结构[](https://rentry.org/llm-training#dataset-structure "Permanent link")

You'll want to outline a structure for your dataset. Essentially, you want:  
您需要为数据集勾勒出一个结构。从本质上讲，您需要：

-   **Data Diversity**: You don't want your models to _only_ do one very specific task. In our assumed use-case, we're training a chat model, but this doesn't mean the data would only be about one _specific_ type of chat/RP. You will want to diversify your training samples, include all kinds of scenarios so that your model can learn how to generate outputs for various types of input.  
    数据多样性：您不希望模型只执行一项非常具体的任务。在我们假设的用例中，我们正在训练一个聊天模型，但这并不意味着数据将只与一种特定类型的聊天/RP 有关。您将希望使训练样本多样化，包括各种场景，以便您的模型可以学习如何为各种类型的输入生成输出。
-   **Dataset size**: Unlike LoRAs or Soft Prompts, you'll want a relatively large amount of data. This is, of course, not on the same level as pre-training datasets. As a rule of thumb, make sure you have **at least** 10 MiB of data for your fine-tune. It's incredibly difficult to overtrain your model, so it's always a good idea to stack more data.  
    数据集大小：与 LoRA 或软提示不同，您需要相对较大的数据量。当然，这与预训练数据集不在同一水平上。根据经验，请确保至少有 10 MiB 的数据用于微调。过度训练模型非常困难，因此堆叠更多数据始终是一个好主意。
-   **Dataset quality**: The quality of your data is incredibly important. You want your dataset to reflect how the model should turn out. If you feed it garbage, it'll spit out garbage.  
    数据集质量：数据质量非常重要。您希望数据集反映模型的结果。如果你给它喂垃圾，它会吐出垃圾。

### Processing the raw dataset  
处理原始数据集[](https://rentry.org/llm-training#processing-the-raw-dataset "Permanent link")

You may have a bunch of text data now. Before you continue, you will want to parse them into a suitable format for pre-processing. Let's assume your dataset is in one of these conditions:  
你现在可能有一堆文本数据。在继续之前，您需要将它们解析为合适的格式以进行预处理。假设您的数据集处于以下条件之一：

##### HTML .HTML[](https://rentry.org/llm-training#html "Permanent link")

You might have HTML files if you scraped your data from websites. In that case, your priority will be pulling data out of the HTML elements. If you're not right in the head, you'll try and use pure RegEx to do this. This is extremely inefficient, so thankfully there are libraries that handle this issue. You can use the [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) Python library to help you with this. You can read the documentation for it, but it's generally used this way:  
如果您从网站上抓取数据，则可能有 HTML 文件。在这种情况下，您的首要任务是从 HTML 元素中提取数据。如果你的头脑不对，你会尝试使用纯正则表达式来做到这一点。这是非常低效的，所以值得庆幸的是，有库可以处理这个问题。您可以使用 Beautiful Soup Python 库来帮助您解决这个问题。您可以阅读它的文档，但它通常以这种方式使用：

<table data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tbody data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tr data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><div data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><pre><span><a href="https://rentry.org/llm-training#L-1-1"> 1</a></span>
<span><a href="https://rentry.org/llm-training#L-1-2"> 2</a></span>
<span><a href="https://rentry.org/llm-training#L-1-3"> 3</a></span>
<span><a href="https://rentry.org/llm-training#L-1-4"> 4</a></span>
<span><a href="https://rentry.org/llm-training#L-1-5"> 5</a></span>
<span><a href="https://rentry.org/llm-training#L-1-6"> 6</a></span>
<span><a href="https://rentry.org/llm-training#L-1-7"> 7</a></span>
<span><a href="https://rentry.org/llm-training#L-1-8"> 8</a></span>
<span><a href="https://rentry.org/llm-training#L-1-9"> 9</a></span>
<span><a href="https://rentry.org/llm-training#L-1-10">10</a></span>
<span><a href="https://rentry.org/llm-training#L-1-11">11</a></span>
<span><a href="https://rentry.org/llm-training#L-1-12">12</a></span>
<span><a href="https://rentry.org/llm-training#L-1-13">13</a></span>
<span><a href="https://rentry.org/llm-training#L-1-14">14</a></span>
<span><a href="https://rentry.org/llm-training#L-1-15">15</a></span>
<span><a href="https://rentry.org/llm-training#L-1-16">16</a></span>
<span><a href="https://rentry.org/llm-training#L-1-17">17</a></span>
<span><a href="https://rentry.org/llm-training#L-1-18">18</a></span>
<span><a href="https://rentry.org/llm-training#L-1-19">19</a></span>
<span><a href="https://rentry.org/llm-training#L-1-20">20</a></span>
<span><a href="https://rentry.org/llm-training#L-1-21">21</a></span>
<span><a href="https://rentry.org/llm-training#L-1-22">22</a></span>
<span><a href="https://rentry.org/llm-training#L-1-23">23</a></span>
<span><a href="https://rentry.org/llm-training#L-1-24">24</a></span>
<span><a href="https://rentry.org/llm-training#L-1-25">25</a></span>
<span><a href="https://rentry.org/llm-training#L-1-26">26</a></span>
<span><a href="https://rentry.org/llm-training#L-1-27">27</a></span>
<span><a href="https://rentry.org/llm-training#L-1-28">28</a></span>
<span><a href="https://rentry.org/llm-training#L-1-29">29</a></span>
<span><a href="https://rentry.org/llm-training#L-1-30">30</a></span>
<span><a href="https://rentry.org/llm-training#L-1-31">31</a></span></pre></div></td><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><div data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><pre><span></span><span id="L-1-1"><a id="L-1-1" name="L-1-1"></a><span>from</span> <span>bs4</span> <span>import</span> <span>BeautifulSoup</span>
</span><span id="L-1-2"><a id="L-1-2" name="L-1-2"></a>
</span><span id="L-1-3"><a id="L-1-3" name="L-1-3"></a><span># HTML content to be parsed</span>
</span><span id="L-1-4"><a id="L-1-4" name="L-1-4"></a><span>html_content</span> <span>=</span> <span>'''</span>
</span><span id="L-1-5"><a id="L-1-5" name="L-1-5"></a><span>&lt;html&gt;</span>
</span><span id="L-1-6"><a id="L-1-6" name="L-1-6"></a><span>&lt;head&gt;</span>
</span><span id="L-1-7"><a id="L-1-7" name="L-1-7"></a><span>&lt;title&gt;Example HTML Page&lt;/title&gt;</span>
</span><span id="L-1-8"><a id="L-1-8" name="L-1-8"></a><span>&lt;/head&gt;</span>
</span><span id="L-1-9"><a id="L-1-9" name="L-1-9"></a><span>&lt;body&gt;</span>
</span><span id="L-1-10"><a id="L-1-10" name="L-1-10"></a><span>&lt;h1&gt;Welcome to the Example Page&lt;/h1&gt;</span>
</span><span id="L-1-11"><a id="L-1-11" name="L-1-11"></a><span>&lt;p&gt;This is a paragraph of text.&lt;/p&gt;</span>
</span><span id="L-1-12"><a id="L-1-12" name="L-1-12"></a><span>&lt;div class="content"&gt;</span>
</span><span id="L-1-13"><a id="L-1-13" name="L-1-13"></a><span>    &lt;h2&gt;Section 1&lt;/h2&gt;</span>
</span><span id="L-1-14"><a id="L-1-14" name="L-1-14"></a><span>    &lt;p&gt;This is the first section.&lt;/p&gt;</span>
</span><span id="L-1-15"><a id="L-1-15" name="L-1-15"></a><span>&lt;/div&gt;</span>
</span><span id="L-1-16"><a id="L-1-16" name="L-1-16"></a><span>&lt;div class="content"&gt;</span>
</span><span id="L-1-17"><a id="L-1-17" name="L-1-17"></a><span>    &lt;h2&gt;Section 2&lt;/h2&gt;</span>
</span><span id="L-1-18"><a id="L-1-18" name="L-1-18"></a><span>    &lt;p&gt;This is the second section.&lt;/p&gt;</span>
</span><span id="L-1-19"><a id="L-1-19" name="L-1-19"></a><span>&lt;/div&gt;</span>
</span><span id="L-1-20"><a id="L-1-20" name="L-1-20"></a><span>&lt;/body&gt;</span>
</span><span id="L-1-21"><a id="L-1-21" name="L-1-21"></a><span>&lt;/html&gt;</span>
</span><span id="L-1-22"><a id="L-1-22" name="L-1-22"></a><span>'''</span>
</span><span id="L-1-23"><a id="L-1-23" name="L-1-23"></a>
</span><span id="L-1-24"><a id="L-1-24" name="L-1-24"></a><span># Create a BeautifulSoup object</span>
</span><span id="L-1-25"><a id="L-1-25" name="L-1-25"></a><span>soup</span> <span>=</span> <span>BeautifulSoup</span><span>(</span><span>html_content</span><span>,</span> <span>'html.parser'</span><span>)</span>
</span><span id="L-1-26"><a id="L-1-26" name="L-1-26"></a>
</span><span id="L-1-27"><a id="L-1-27" name="L-1-27"></a><span># Extract text from the HTML</span>
</span><span id="L-1-28"><a id="L-1-28" name="L-1-28"></a><span>text</span> <span>=</span> <span>soup</span><span>.</span><span>get_text</span><span>()</span>
</span><span id="L-1-29"><a id="L-1-29" name="L-1-29"></a>
</span><span id="L-1-30"><a id="L-1-30" name="L-1-30"></a><span># Print the extracted text</span>
</span><span id="L-1-31"><a id="L-1-31" name="L-1-31"></a><span>print</span><span>(</span><span>text</span><span>)</span>
</span></pre></div></td></tr></tbody></table>

You'll have an output like this:  
您将获得如下输出：

<table data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tbody data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tr data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"></td><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><div data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><pre><span></span><span id="L-2-1"><a id="L-2-1" name="L-2-1"></a>Example HTML Page
</span><span id="L-2-2"><a id="L-2-2" name="L-2-2"></a>
</span><span id="L-2-3"><a id="L-2-3" name="L-2-3"></a>Welcome to the Example Page
</span><span id="L-2-4"><a id="L-2-4" name="L-2-4"></a>This is a paragraph of text.
</span><span id="L-2-5"><a id="L-2-5" name="L-2-5"></a>Section 1
</span><span id="L-2-6"><a id="L-2-6" name="L-2-6"></a>This is the first section.
</span><span id="L-2-7"><a id="L-2-7" name="L-2-7"></a>Section 2
</span><span id="L-2-8"><a id="L-2-8" name="L-2-8"></a>This is the second section.
</span></pre></div></td></tr></tbody></table>

##### CSV[](https://rentry.org/llm-training#csv "Permanent link")

You could have CSV files if you've obtained your dataset from online open data sources. The easiest way to parse them is using the `pandas` python library. The basic usage would be:  
如果从联机开放数据源获取数据集，则可以使用 CSV 文件。解析它们的最简单方法是使用 `pandas` python 库。基本用法是：

<table data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tbody data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tr data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"></td><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><div data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><pre><span></span><span id="L-3-1"><a id="L-3-1" name="L-3-1"></a><span>import</span> <span>pandas</span> <span>as</span> <span>pd</span>
</span><span id="L-3-2"><a id="L-3-2" name="L-3-2"></a>
</span><span id="L-3-3"><a id="L-3-3" name="L-3-3"></a><span># Read the CSV file</span>
</span><span id="L-3-4"><a id="L-3-4" name="L-3-4"></a><span>df</span> <span>=</span> <span>pd</span><span>.</span><span>read_csv</span><span>(</span><span>'dataset.csv'</span><span>)</span>
</span><span id="L-3-5"><a id="L-3-5" name="L-3-5"></a>
</span><span id="L-3-6"><a id="L-3-6" name="L-3-6"></a><span># Extract plaintext from a specific column</span>
</span><span id="L-3-7"><a id="L-3-7" name="L-3-7"></a><span>column_data</span> <span>=</span> <span>df</span><span>[</span><span>'column_name'</span><span>]</span><span>.</span><span>astype</span><span>(</span><span>str</span><span>)</span>
</span><span id="L-3-8"><a id="L-3-8" name="L-3-8"></a><span>plaintext</span> <span>=</span> <span>column_data</span><span>.</span><span>to_string</span><span>(</span><span>index</span><span>=</span><span>False</span><span>)</span>
</span><span id="L-3-9"><a id="L-3-9" name="L-3-9"></a>
</span><span id="L-3-10"><a id="L-3-10" name="L-3-10"></a><span># Print the extracted plaintext data</span>
</span><span id="L-3-11"><a id="L-3-11" name="L-3-11"></a><span>print</span><span>(</span><span>plaintext</span><span>)</span>
</span></pre></div></td></tr></tbody></table>

You'll have to specify the column name.  
您必须指定列名称。

##### SQL[](https://rentry.org/llm-training#sql "Permanent link")

This one will be a bit tougher. You can take the sensible approach and use a DB framework such as MariaDB or PostgreSQL to parse the dataset into plaintext, but there are also Python libraries for this purpose; one example is [sqlparse](https://sqlparse.readthedocs.io/en/latest/). The basic usage is:  
这个会有点艰难。您可以采取明智的方法，使用 MariaDB 或 PostgreSQL 等数据库框架将数据集解析为明文，但也有用于此目的的 Python 库;一个例子是 SQLPARSE。基本用法是：

<table data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tbody data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tr data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><div data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><pre><span><a href="https://rentry.org/llm-training#L-4-1"> 1</a></span>
<span><a href="https://rentry.org/llm-training#L-4-2"> 2</a></span>
<span><a href="https://rentry.org/llm-training#L-4-3"> 3</a></span>
<span><a href="https://rentry.org/llm-training#L-4-4"> 4</a></span>
<span><a href="https://rentry.org/llm-training#L-4-5"> 5</a></span>
<span><a href="https://rentry.org/llm-training#L-4-6"> 6</a></span>
<span><a href="https://rentry.org/llm-training#L-4-7"> 7</a></span>
<span><a href="https://rentry.org/llm-training#L-4-8"> 8</a></span>
<span><a href="https://rentry.org/llm-training#L-4-9"> 9</a></span>
<span><a href="https://rentry.org/llm-training#L-4-10">10</a></span>
<span><a href="https://rentry.org/llm-training#L-4-11">11</a></span>
<span><a href="https://rentry.org/llm-training#L-4-12">12</a></span>
<span><a href="https://rentry.org/llm-training#L-4-13">13</a></span>
<span><a href="https://rentry.org/llm-training#L-4-14">14</a></span>
<span><a href="https://rentry.org/llm-training#L-4-15">15</a></span>
<span><a href="https://rentry.org/llm-training#L-4-16">16</a></span>
<span><a href="https://rentry.org/llm-training#L-4-17">17</a></span>
<span><a href="https://rentry.org/llm-training#L-4-18">18</a></span>
<span><a href="https://rentry.org/llm-training#L-4-19">19</a></span></pre></div></td><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><div data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><pre><span></span><span id="L-4-1"><a id="L-4-1" name="L-4-1"></a><span>&gt;&gt;&gt;</span> <span>import</span> <span>sqlparse</span>
</span><span id="L-4-2"><a id="L-4-2" name="L-4-2"></a>
</span><span id="L-4-3"><a id="L-4-3" name="L-4-3"></a><span>&gt;&gt;&gt;</span> <span># Split a string containing two SQL statements:</span>
</span><span id="L-4-4"><a id="L-4-4" name="L-4-4"></a><span>&gt;&gt;&gt;</span> <span>raw</span> <span>=</span> <span>'select * from foo; select * from bar;'</span>
</span><span id="L-4-5"><a id="L-4-5" name="L-4-5"></a><span>&gt;&gt;&gt;</span> <span>statements</span> <span>=</span> <span>sqlparse</span><span>.</span><span>split</span><span>(</span><span>raw</span><span>)</span>
</span><span id="L-4-6"><a id="L-4-6" name="L-4-6"></a><span>&gt;&gt;&gt;</span> <span>statements</span>
</span><span id="L-4-7"><a id="L-4-7" name="L-4-7"></a><span>[</span><span>'select * from foo;'</span><span>,</span> <span>'select * from bar;'</span><span>]</span>
</span><span id="L-4-8"><a id="L-4-8" name="L-4-8"></a>
</span><span id="L-4-9"><a id="L-4-9" name="L-4-9"></a><span>&gt;&gt;&gt;</span> <span># Format the first statement and print it out:</span>
</span><span id="L-4-10"><a id="L-4-10" name="L-4-10"></a><span>&gt;&gt;&gt;</span> <span>first</span> <span>=</span> <span>statements</span><span>[</span><span>0</span><span>]</span>
</span><span id="L-4-11"><a id="L-4-11" name="L-4-11"></a><span>&gt;&gt;&gt;</span> <span>print</span><span>(</span><span>sqlparse</span><span>.</span><span>format</span><span>(</span><span>first</span><span>,</span> <span>reindent</span><span>=</span><span>True</span><span>,</span> <span>keyword_case</span><span>=</span><span>'upper'</span><span>))</span>
</span><span id="L-4-12"><a id="L-4-12" name="L-4-12"></a><span>SELECT</span> <span>*</span>
</span><span id="L-4-13"><a id="L-4-13" name="L-4-13"></a><span>FROM</span> <span>foo</span><span>;</span>
</span><span id="L-4-14"><a id="L-4-14" name="L-4-14"></a>
</span><span id="L-4-15"><a id="L-4-15" name="L-4-15"></a><span>&gt;&gt;&gt;</span> <span># Parsing a SQL statement:</span>
</span><span id="L-4-16"><a id="L-4-16" name="L-4-16"></a><span>&gt;&gt;&gt;</span> <span>parsed</span> <span>=</span> <span>sqlparse</span><span>.</span><span>parse</span><span>(</span><span>'select * from foo'</span><span>)[</span><span>0</span><span>]</span>
</span><span id="L-4-17"><a id="L-4-17" name="L-4-17"></a><span>&gt;&gt;&gt;</span> <span>parsed</span><span>.</span><span>tokens</span>
</span><span id="L-4-18"><a id="L-4-18" name="L-4-18"></a><span>[</span><span>&lt;</span><span>DML</span> <span>'select'</span> <span>at</span> <span>0x7f22c5e15368</span><span>&gt;</span><span>,</span> <span>&lt;</span><span>Whitespace</span> <span>' '</span> <span>at</span> <span>0x7f22c5e153b0</span><span>&gt;</span><span>,</span> <span>&lt;</span><span>Wildcard</span> <span>'*'</span> <span>…</span> <span>]</span>
</span><span id="L-4-19"><a id="L-4-19" name="L-4-19"></a><span>&gt;&gt;&gt;</span>
</span></pre></div></td></tr></tbody></table>

### Minimizing the noise 将噪音降至最低[](https://rentry.org/llm-training#minimizing-the-noise "Permanent link")

The best language models are stochastic, which makes it difficult to predict their behaviour, even if the input prompt remains the same. This can, on occasion, result in low-quality and undesirable outputs. You will want to make sure your dataset is cleaned out of unwanted elements. This is doubly important if your data source is synthetic, i.e. generating by GPT-4/3. You might want to truncate or remove the mention of phrases such as "As an AI language model...", "harmful or offensive content...", "...trained by OpenAI...", etc. [This script](https://huggingface.co/datasets/ehartford/wizard_vicuna_70k_unfiltered/blob/main/optional_clean.py) by [ehartford](https://huggingface.co/ehartford) is a good filter for this specific task. You can also refer to the [gptslop](https://github.com/AlpinDale/gptslop) repo.  
最好的语言模型是随机的，这使得很难预测它们的行为，即使输入提示保持不变。这有时会导致低质量和不良输出。您需要确保清除数据集中不需要的元素。如果您的数据源是合成的，即由 GPT-4/3 生成，这一点就更加重要。您可能希望截断或删除提及的短语，例如“作为 AI 语言模型...”、“有害或冒犯性内容...”、“......由 OpenAI 训练......“等。ehartford 的这个脚本是这个特定任务的一个很好的过滤器。您也可以参考 gptslop 存储库。

### Starting the training run  
开始训练运行[](https://rentry.org/llm-training#starting-the-training-run "Permanent link")

We will use the [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) trainer for fine-tuning, as it's simple to use and has all the features we need.  
我们将使用蝾螈训练器进行微调，因为它使用简单并且具有我们需要的所有功能。

If you're using cloud computing services, like RunPod, you likely have all the requirements necessary.  
如果您使用的是云计算服务，例如 RunPod，您可能满足所有必要的要求。

1.  Clone the repository and install requirements:  
    克隆存储库和安装要求：  
    
    <table data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tbody data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tr data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"></td><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><div data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><pre><span></span><span id="L-5-1"><a id="L-5-1" name="L-5-1"></a>git clone https://github.com/OpenAccess-AI-Collective/axolotl <span>&amp;&amp;</span> <span>cd</span> axolotl
    </span><span id="L-5-2"><a id="L-5-2" name="L-5-2"></a>
    </span><span id="L-5-3"><a id="L-5-3" name="L-5-3"></a>pip3 install packaging
    </span><span id="L-5-4"><a id="L-5-4" name="L-5-4"></a>pip3 install -e <span>'.[flash-attn,deepspeed]'</span>
    </span></pre></div></td></tr></tbody></table>
    

This will install axolotl, and then we're ready to begin finetuning.  
这将安装蝾螈，然后我们就可以开始微调了。

Axolotl takes all the options for training in a single `yaml` file. There are already some sample configs in the `examples` directory, for various different models.  
蝾螈将所有训练选项都放在一个 `yaml` 文件中。 `examples` 目录中已经有一些示例配置，适用于各种不同的模型。

For this example, we'll train the Mistral model using QLoRA method, which should make it possible on a single 3090 GPU. To start the run, simply execute this command:  
在此示例中，我们将使用 QLoRA 方法训练 Mistral 模型，这应该可以在单个 3090 GPU 上实现。要开始运行，只需执行以下命令：

<table data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tbody data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tr data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"></td><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><div data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><pre><span></span><span id="L-6-1"><a id="L-6-1" name="L-6-1"></a>accelerate launch -m axolotl.cli.train examples/mistral/config.yml
</span></pre></div></td></tr></tbody></table>

And congrats! You just trained Mistral! The example config uses a very small dataset, which should take very little time to train on.  
恭喜！你刚刚训练了米斯特拉尔！示例配置使用一个非常小的数据集，该数据集应该花费很少的时间来训练。

To use a custom dataset, you'll want to properly format it into a `JSONL` file. Axolotl takes many different formats, you can find examples [here](https://github.com/OpenAccess-AI-Collective/axolotl#dataset). You can then edit the `qlora.yml` file and point it to your dataset. The full explanation for all the config options are [here](https://github.com/OpenAccess-AI-Collective/axolotl#config), make sure you click on the expand button to see all of them!  
若要使用自定义数据集，需要将其正确格式化为 `JSONL` 文件。蝾螈采用许多不同的格式，您可以在此处找到示例。然后， `qlora.yml` 您可以编辑文件并将其指向数据集。所有配置选项的完整说明都在这里，请确保单击展开按钮以查看所有选项！

You know how to train a model now, but let's go through some very important info in the next sections. We'll begin with explaining what LoRA actually is, and why it's effective.  
您现在知道如何训练模型，但让我们在接下来的部分中介绍一些非常重要的信息。我们将首先解释 LoRA 到底是什么， 以及为什么它有效.

___

## Low-Rank Adaptation (LoRA)  
低秩适应 （LoRA）[](https://rentry.org/llm-training#low-rank-adaptation-lora_1 "Permanent link")

LoRA is a training method designed to expedite the training process of large language models, all while reducing memory consumption. By introducing pairs of rank-decomposition weight matrices, known as update matrices, to the existing weights, LoRA focuses solely on training these new added weights. This approach offers several advantages:  
LoRA 是一种训练方法，旨在加快大型语言模型的训练过程，同时减少内存消耗。通过引入成对的秩分解权重矩阵， 称为更新矩阵， LoRA 只专注于训练这些新添加的权重.这种方法具有以下几个优点：

1.  Preservation of pretrained Weights: LoRA maintains the frozen state of previously trained weights, minimizing the risk of catastrophic forgetting. This ensures that the model retains its existing knowledge while adapting to new data.  
    保留预训练的权重： LoRA 保持先前训练的权重的冻结状态， 将灾难性遗忘的风险降至最低.这确保了模型在适应新数据的同时保留其现有知识。
2.  Portability of trained weights: The rank-decomposition matrices used in LoRA have significantly fewer parameters compared to the original model. This characteristic allows the trained LoRA weights to be easily transferred and utilized in other contexts, making them highly portable.  
    训练权重的可移植性： 与原始模型相比，LoRA 中使用的秩分解矩阵的参数要少得多.这一特性使经过训练的 LoRA 权重能够轻松转移并在其他环境中使用， 使其具有高度的便携性.
3.  Integration with Attention Layers: LoRA matrices are typically incorporated into the attention layers of the original model. Additionally, the adaptation scale parameter allows control over the extent to which the model adjusts to new training data.  
    与注意力层集成：LoRA 矩阵通常合并到原始模型的注意力层中。此外，适应性尺度参数允许控制模型适应新训练数据的程度。
4.  Memory efficiency: LoRA's improved memory efficiency opens up the possibily of running fine-tune tasks on less than 3x the required compute for a native fine-tune.  
    内存效率： LoRA 改进的内存效率开辟了在不到本机微调所需计算量 3 倍的情况下运行微调任务的可能性.

### LoRA hyperparameters LoRA 超参数[](https://rentry.org/llm-training#lora-hyperparameters "Permanent link")

#### LoRA Rank LoRA排名[](https://rentry.org/llm-training#lora-rank "Permanent link")

This determines the number of rank decomposition matrices. Rank decomposition is applied to weight matrices in order to reduce memory consumption and computational requirements. The original [LoRA paper](https://arxiv.org/pdf/2106.09685.pdf) recommends a rank of 8 (`r = 8`) as the minimum amount. Keep in mind that higher ranks lead to better results and higher compute requirements. The more complex your dataset, the higher your rank will need to be.  
这决定了秩分解矩阵的数量。秩分解应用于权重矩阵，以减少内存消耗和计算要求。原始 LoRA 论文建议将 8 （ `r = 8` ） 作为最低等级。请记住，排名越高，结果越好，计算要求越高。数据集越复杂，排名就越高。

To match a full fine-tune, you can set the rank to equal to the model's hidden size. This is, however, not recommended because it's a massive waste of resources. You can find out the model's hidden size by reading through the `config.json` or by loading the model with [Transformers](https://github.com/huggingface/transformers)'s `AutoModel` and using the `model.config.hidden_size` function:  
要匹配完全微调，您可以将秩设置为等于模型的隐藏大小。但是，不建议这样做，因为这会浪费大量资源。您可以通过通读 `config.json` 或使用 Transformer 加载模型 `AutoModel` 并使用以下 `model.config.hidden_size` 函数来找出模型的隐藏大小：  

<table data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tbody data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tr data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"></td><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><div data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><pre><span></span><span id="L-7-1"><a id="L-7-1" name="L-7-1"></a><span>from</span> <span>transformers</span> <span>import</span> <span>AutoModelForCausalLM</span>
</span><span id="L-7-2"><a id="L-7-2" name="L-7-2"></a><span>model_name</span> <span>=</span> <span>"huggyllama/llama-7b"</span>      <span># can also be a local directory</span>
</span><span id="L-7-3"><a id="L-7-3" name="L-7-3"></a><span>model</span> <span>=</span> <span>AutoModelForCausalLM</span><span>.</span><span>from_pretrained</span><span>(</span><span>model_name</span><span>)</span>
</span><span id="L-7-4"><a id="L-7-4" name="L-7-4"></a><span>hidden_size</span> <span>=</span> <span>model</span><span>.</span><span>config</span><span>.</span><span>hidden_size</span>
</span><span id="L-7-5"><a id="L-7-5" name="L-7-5"></a><span>print</span><span>(</span><span>hidden_size</span><span>)</span>
</span></pre></div></td></tr></tbody></table>

#### LoRA Alpha LoRA 阿尔法[](https://rentry.org/llm-training#lora-alpha "Permanent link")

I'm not 100% sure about this :D  
我不是 100% 确定这个:D

This is the scaling factor for the LoRA, which determines the extent to which the model is adapted towards new training data. The alpha value adjusts the contribution of the update matrices during the train process. Lower values give more weight to the original data and maintain the model's existing knowledge to a greater extent than higher values.  
这是 LoRA 的比例因子，它决定了模型适应新训练数据的程度。alpha 值在训练过程中调整更新矩阵的贡献。与较高的值相比，较低的值赋予原始数据更多的权重，并在更大程度上保持模型的现有知识。

#### LoRA Target Modules LoRA 目标模块[](https://rentry.org/llm-training#lora-target-modules "Permanent link")

Here you can determine which specific weights and matrices are to be trained. The most basic ones to train are the Query Vectors (e.g. `q_proj`) and Value Vectors (e.g. `v_proj`) projection matrices. The names of these matrices will differ from model to model. You can find out the exact names by running the following script:  
在这里，您可以确定要训练哪些特定的权重和矩阵。最基本的训练是查询向量（例如）和价值向量（例如 `q_proj` `v_proj` ）投影矩阵。这些矩阵的名称因模型而异。您可以通过运行以下脚本来查找确切的名称：

<table data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tbody data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tr data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"></td><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><div data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><pre><span></span><span id="L-8-1"><a id="L-8-1" name="L-8-1"></a><span>from</span> <span>transformers</span> <span>import</span> <span>AutoModelForCausalLM</span>
</span><span id="L-8-2"><a id="L-8-2" name="L-8-2"></a><span>model_name</span> <span>=</span> <span>"huggyllama/llama-7b"</span>      <span># can also be a local directory</span>
</span><span id="L-8-3"><a id="L-8-3" name="L-8-3"></a><span>model</span> <span>=</span> <span>AutoModelForCausalLM</span><span>.</span><span>from_pretrained</span><span>(</span><span>model_name</span><span>)</span>
</span><span id="L-8-4"><a id="L-8-4" name="L-8-4"></a><span>layer_names</span> <span>=</span> <span>model</span><span>.</span><span>state_dict</span><span>()</span><span>.</span><span>keys</span><span>()</span>
</span><span id="L-8-5"><a id="L-8-5" name="L-8-5"></a>
</span><span id="L-8-6"><a id="L-8-6" name="L-8-6"></a><span>for</span> <span>name</span> <span>in</span> <span>layer_names</span><span>:</span>
</span><span id="L-8-7"><a id="L-8-7" name="L-8-7"></a>    <span>print</span><span>(</span><span>name</span><span>)</span>
</span></pre></div></td></tr></tbody></table>

This will give you an output like this:  
这将为您提供如下输出：  

<table data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tbody data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><tr data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><div data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><pre><span><a href="https://rentry.org/llm-training#L-9-1"> 1</a></span>
<span><a href="https://rentry.org/llm-training#L-9-2"> 2</a></span>
<span><a href="https://rentry.org/llm-training#L-9-3"> 3</a></span>
<span><a href="https://rentry.org/llm-training#L-9-4"> 4</a></span>
<span><a href="https://rentry.org/llm-training#L-9-5"> 5</a></span>
<span><a href="https://rentry.org/llm-training#L-9-6"> 6</a></span>
<span><a href="https://rentry.org/llm-training#L-9-7"> 7</a></span>
<span><a href="https://rentry.org/llm-training#L-9-8"> 8</a></span>
<span><a href="https://rentry.org/llm-training#L-9-9"> 9</a></span>
<span><a href="https://rentry.org/llm-training#L-9-10">10</a></span>
<span><a href="https://rentry.org/llm-training#L-9-11">11</a></span>
<span><a href="https://rentry.org/llm-training#L-9-12">12</a></span>
<span><a href="https://rentry.org/llm-training#L-9-13">13</a></span>
<span><a href="https://rentry.org/llm-training#L-9-14">14</a></span>
<span><a href="https://rentry.org/llm-training#L-9-15">15</a></span>
<span><a href="https://rentry.org/llm-training#L-9-16">16</a></span></pre></div></td><td data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><div data-immersive-translate-walked="3ad58452-b520-4e0b-860b-5ec8f5b7726e"><pre><span></span><span id="L-9-1"><a id="L-9-1" name="L-9-1"></a><span>model</span><span>.</span><span>embed_tokens</span><span>.</span><span>weight</span>
</span><span id="L-9-2"><a id="L-9-2" name="L-9-2"></a><span>model</span><span>.</span><span>layers</span><span>.0</span><span>.</span><span>self_attn</span><span>.</span><span>q_proj</span><span>.</span><span>weight</span>
</span><span id="L-9-3"><a id="L-9-3" name="L-9-3"></a><span>model</span><span>.</span><span>layers</span><span>.0</span><span>.</span><span>self_attn</span><span>.</span><span>k_proj</span><span>.</span><span>weight</span>
</span><span id="L-9-4"><a id="L-9-4" name="L-9-4"></a><span>model</span><span>.</span><span>layers</span><span>.0</span><span>.</span><span>self_attn</span><span>.</span><span>v_proj</span><span>.</span><span>weight</span>
</span><span id="L-9-5"><a id="L-9-5" name="L-9-5"></a><span>model</span><span>.</span><span>layers</span><span>.0</span><span>.</span><span>self_attn</span><span>.</span><span>o_proj</span><span>.</span><span>weight</span>
</span><span id="L-9-6"><a id="L-9-6" name="L-9-6"></a><span>model</span><span>.</span><span>layers</span><span>.0</span><span>.</span><span>self_attn</span><span>.</span><span>rotary_emb</span><span>.</span><span>inv_freq</span>
</span><span id="L-9-7"><a id="L-9-7" name="L-9-7"></a><span>model</span><span>.</span><span>layers</span><span>.0</span><span>.</span><span>mlp</span><span>.</span><span>gate_proj</span><span>.</span><span>weight</span>
</span><span id="L-9-8"><a id="L-9-8" name="L-9-8"></a><span>model</span><span>.</span><span>layers</span><span>.0</span><span>.</span><span>mlp</span><span>.</span><span>down_proj</span><span>.</span><span>weight</span>
</span><span id="L-9-9"><a id="L-9-9" name="L-9-9"></a><span>model</span><span>.</span><span>layers</span><span>.0</span><span>.</span><span>mlp</span><span>.</span><span>up_proj</span><span>.</span><span>weight</span>
</span><span id="L-9-10"><a id="L-9-10" name="L-9-10"></a><span>model</span><span>.</span><span>layers</span><span>.0</span><span>.</span><span>input_layernorm</span><span>.</span><span>weight</span>
</span><span id="L-9-11"><a id="L-9-11" name="L-9-11"></a><span>model</span><span>.</span><span>layers</span><span>.0</span><span>.</span><span>post_attention_layernorm</span><span>.</span><span>weight</span>
</span><span id="L-9-12"><a id="L-9-12" name="L-9-12"></a>
</span><span id="L-9-13"><a id="L-9-13" name="L-9-13"></a><span>...</span>
</span><span id="L-9-14"><a id="L-9-14" name="L-9-14"></a>
</span><span id="L-9-15"><a id="L-9-15" name="L-9-15"></a><span>model</span><span>.</span><span>norm</span><span>.</span><span>weight</span>
</span><span id="L-9-16"><a id="L-9-16" name="L-9-16"></a><span>lm_head</span><span>.</span><span>weight</span>
</span></pre></div></td></tr></tbody></table>

The naming convention is essentially: `{identifier}.{layer}.{layer_number}.{component}.{module}.{parameter}`. Here's a basic explanation for each module (keep in mind that these names are different for each model architecture):  
命名约定实质上是： `{identifier}.{layer}.{layer_number}.{component}.{module}.{parameter}` .以下是每个模块的基本说明（请记住，这些名称对于每个模型体系结构都不同）：

-   `up_proj`: The projection matrix used in the upward (decoder to encoder) attention pass. It projects the decoder's hidden states to the same dimension as the encoder's hidden states for compatibility during attention calculations.  
    `up_proj` ：向上（解码器到编码器）注意力传递中使用的投影矩阵。它将解码器的隐藏状态投影到与编码器隐藏状态相同的维度，以便在注意力计算期间实现兼容性。
-   `down_proj`: The projection matrix used in the downward (encoder to decoder) attention pass. It projects the encoder's hidden states to the dimension expected by thr decoder for attention calculations.  
    `down_proj` ：向下（编码器到解码器）注意力传递中使用的投影矩阵。它将编码器的隐藏状态投影到解码器预期的维度，以便进行注意力计算。
-   `q_proj`: The projection matrix applied to the query vectors in the attention mechanism. Transforms the input hidden states to the desired dimension for effective query representations.  
    `q_proj` ：应用于注意力机制中查询向量的投影矩阵。将输入隐藏状态转换为所需的维度，以实现有效的查询表示形式。
-   `v_proj`: The projection matrix applied to the value vectors in the attention mechanism. Transforms the input hidden states to the desired dimension for effective value representations.  
    `v_proj` ：应用于注意力机制中值向量的投影矩阵。将输入的隐藏状态转换为所需的维度，以实现有效的值表示。
-   `k_proj`: The projection matrix applied to the key vectors blah blah.  
    `k_proj` ：应用于关键向量的投影矩阵等等。
-   `o_proj`: The projection matrix applied to the output of the attention mechanism. Transforms the combined attention output to the desired dimension before further processing.  
    `o_proj` ：应用于注意力机制输出的投影矩阵。在进一步处理之前，将组合的注意力输出转换为所需的维度。

There are, however, three (or 4, if your model has biases) outliers. They do not follow the naming convention specified above, foregoing the layer name and number. These are:  
但是，有三个（或四个，如果您的模型有偏差）异常值。它们不遵循上面指定的命名约定，放弃了图层名称和编号。这些是：

-   **Embedding Token Weights** `embed_tokens`: Represents the params associated with the embedding layer of the model, usually placed at the beginning of the model as it serves to map input tokens or words to their corresponding dense vector representations. **Important to target if your dataset has custom syntax.**  
    嵌入标记权重 `embed_tokens` ：表示与模型嵌入层关联的参数，通常放置在模型的开头，因为它用于将输入标记或单词映射到其相应的密集向量表示。如果数据集具有自定义语法，则目标非常重要。
-   **Normalization Weights** `norm`: The normalization layer within the model. Layer or batch normalizations are often used to improve the stability and converge of deep neural networks. These are typically placed within or after certain layers in the model's architecture to mitigate issues like vanishing or exploding gradients and to aid in faster training and better generalization. Generally not targeted for LoRA.  
    归一化权重 `norm` ：模型中的归一化层。层或批量归一化通常用于提高深度神经网络的稳定性和收敛性。它们通常放置在模型架构中的某些层内或之后，以缓解梯度消失或爆炸等问题，并帮助加快训练速度和实现更好的泛化。通常不针对 LoRA。
-   **Output Layer** `lm_head`: The output layer of a language modeling LLM. It's responsible for generating predictions or scores for the next token based on the learned representations from the _preceding_ layers. Placed at the bottom. **Important to target if your dataset has custom syntax.**  
    输出层：语言建模LLM的输出层 `lm_head` 。它负责根据从前几层学习到的表示形式为下一个令牌生成预测或分数。放置在底部。如果数据集具有自定义语法，则目标非常重要。

## QLoRA QLoRA的[](https://rentry.org/llm-training#qlora "Permanent link")

QLoRA (Quantized Low Rank Adapters) is an efficient finetuning approach that reduces memory usage while maintaining high performance for large language models. It enables the finetuning of a 65B parameter model on a single 48GB GPU, while preserving full 16-bit fine-tuning task performance.  
QLoRA（量化低秩适配器）是一种高效的微调方法，可减少内存使用量，同时保持大型语言模型的高性能。它支持在单个 48GB GPU 上对 65B 参数模型进行微调，同时保留完整的 16 位微调任务性能。

The key innovations of QLoRA include:  
QLoRA 的主要创新包括：

-   Backpropagation of gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA).  
    通过冻结的 4 位量化预训练语言模型将梯度反向传播到低秩适配器 （LoRA） 中。
-   Use of a new data type called 4-bit NormalFloat (NF4), which optimally handles normally distributed weights.  
    使用称为 4 位 NormalFloat （NF4） 的新数据类型，该数据类型以最佳方式处理正态分布的权重。
-   Double quantization to reduce the average memory footprint by quantizing the quantization constants.
-   Paged optimizers to effectively manage memory spikes during the finetuning process.  
    分页优化器，用于在微调过程中有效管理内存峰值。

In the next sections, we'll try and go through what all the training hyperparameters, aka configs, do.  
在接下来的章节中，我们将尝试介绍所有训练超参数（也称为配置）的作用。

## Training Hyperparameters 训练超参数[](https://rentry.org/llm-training#training-hyperparameters "Permanent link")

Training hyperparameters play a crucial role in shaping the behaviour and performance of your models. These hparams are settings that guide the training process, determining how the model learns from the provided data. Selecting appropriate hparams can significantly impact the model's convergence, generalization, and overall effectiveness.  
训练超参数在塑造模型的行为和性能方面起着至关重要的作用。这些 hparam 是指导训练过程的设置，用于确定模型如何从提供的数据中学习。选择适当的 hparam 可以显著影响模型的收敛性、泛化性和整体有效性。

In this section, we'll try and explain the key training hparams that require careful consideration during the training phase. We'll go over concepts like batch size, epochs, learning rate, regularization, and more. By gaining a deep understanding of these hparams and their effects, you will be equipped to fine-tune and optimize your models effectively, ensuring optimal performance in various machine learning tasks. So let's dive in and unravel the mysteries behind training hparams.  
在本节中，我们将尝试解释在训练阶段需要仔细考虑的关键训练参数。我们将讨论批量大小、纪元、学习率、正则化等概念。通过深入了解这些参数及其效果，您将能够有效地微调和优化模型，确保在各种机器学习任务中获得最佳性能。因此，让我们深入了解并解开训练 hparams 背后的奥秘。

### Batch Size and Epoch 批量大小和纪元[](https://rentry.org/llm-training#batch-size-and-epoch "Permanent link")

Stochastic gradient descent (SGD) is a learning algorithm with multiple hyperparameters for use. Two that often confuse a novice are the batch size and number of epochs. They're both **integer** values and seemingly do the same thing. Let's go over the main outtakes for this section:  
随机梯度下降 （SGD） 是一种具有多个超参数的学习算法。经常让新手感到困惑的两个是批量大小和 epoch 数量。它们都是整数值，似乎做同样的事情。让我们回顾一下本节的主要内容：

-   **Stochastic Gradient Descent (SGD):** It is an iterative learning algorithm that utilizes a training dataset to update a model gradually.  
    随机梯度下降 （SGD）：它是一种迭代学习算法，它利用训练数据集逐步更新模型。
-   **Batch Size**: The batch size is a hyperparameter in gradient descent that determines the number of training samples processed before updating the model's internal parameters. In other words, it specifies how many samples are used in each iteration to calculate the error and adjust the model.  
    批量大小：批量大小是梯度下降中的超参数，用于确定在更新模型内部参数之前处理的训练样本数。换句话说，它指定在每次迭代中使用多少样本来计算误差和调整模型。
-   **Number of Epochs**: The number of epochs is another hyperparameter in gradient descent, which controls the number of complete passes through the training dataset. Each epoch involves processing the entire dataset once, and the model's parameters are updated after every epoch.  
    周期数：周期数是梯度下降中的另一个超参数，它控制通过训练数据集的完整传递次数。每个 epoch 涉及处理整个数据集一次，模型的参数在每个 epoch 之后更新。

We'll have to divide this section into five (5) parts.  
我们必须将本节分为五 （5） 个部分。

#### Stochastic Gradient Descent  
随机梯度下降[](https://rentry.org/llm-training#stochastic-gradient-descent "Permanent link")

Stochastic Gradient Descent (SGD) is an optimization algorithm used to find the best internal parameters for a model, aiming to minimize performance measures like logarithmic loss or mean squared error. For a detailed overview of these measures, you can refer to [this article](https://towardsdatascience.com/understanding-the-3-most-common-loss-functions-for-machine-learning-regression-23e0ef3e14d3).  
随机梯度下降 （SGD） 是一种优化算法，用于查找模型的最佳内部参数，旨在最大限度地减少对数损失或均方误差等性能度量。有关这些措施的详细概述，可以参考此文章。

Optimization can be thought of as a search process, where the algorithm learns to improve the model. The specific optimization algorithm used is called "gradient descent." Here, "gradient" refers to calculating the error slope or [gradient](https://en.wikipedia.org/wiki/Gradient), while "descent" signifies moving downwards along this slope to approach a minimal error level.  
优化可以被认为是一个搜索过程，算法在其中学习以改进模型。使用的特定优化算法称为“梯度下降”。在这里，“梯度”是指计算误差斜率或梯度，而“下降”是指沿着该斜率向下移动以接近最小误差水平。

The algorithm works iteratively, meaning it goes through multiple discrete steps, with each step aiming to enhance the model parameters. During each step, the model uses the current set of internal parameters to make predictions on a subset of samples. These predictions are then compared to the actual expected outcomes, allowing for the calculation of an error. This error is then utilized to update the internal model parameters. The update procedure varies depending on the algorithm being used, but in the case of artificial neural networks, the backpropagation update algorithm is employed.  
该算法以迭代方式工作，这意味着它经历了多个离散步骤，每个步骤都旨在增强模型参数。在每个步骤中，模型使用当前的内部参数集对样本子集进行预测。然后将这些预测与实际的预期结果进行比较，从而计算误差。然后利用此错误更新内部模型参数。更新过程因所使用的算法而异，但在人工神经网络的情况下，采用反向传播更新算法。

Before delving into the concepts of batches and epochs, let's clarify what we mean by a "sample."  
在深入研究批次和纪元的概念之前，让我们先澄清一下“样本”的含义。

#### Sample[](https://rentry.org/llm-training#sample "Permanent link")

A sample or a sequence is a single row of data. It contains inputs that are fed into the algorithm and an output that is used to compare to the prediction and calculate an error.

A training dataset is comprised of many rows of data, e.g. many samples. A sample may also be called an instance, an observation, an input vector, a sequence, or a feature vector.

Now that we know what a sample is, let's define a **batch**.

#### Batch[](https://rentry.org/llm-training#batch "Permanent link")

The batch size is a hparam that determines how many samples are processed before updating the model's internal parameters. Imagine a "for-loop" that iterates through a specific number of samples and makes predictions. After processing the batch, the predictions are compared to the expected outputs, and an error is calculated. This error is then used to improve the model by adjusting its parameters, moving in the direction of the error gradient.

A training dataset can be divided into one or more batches. Here are the different types of gradient descent algorithms based on batch sizes:

-   Batch Gradient Descent: When the batch size is equal to the total number of training samples, it is called batch gradient descent. In this case, the entire dataset is used to compute predictions and calculate the error before updating the model.
-   Stochastic Gradient Descent: When the batch size is set to one, it is referred to as stochastic gradient descent. Here, each sample is processed individually, and the model parameters are updated after each sample. This approach introduces more randomness into the learning process.
-   Mini-Batch Gradient Descent: When the batch size is greater than one and less than the total size of the training dataset, it is called mini-batch gradient descent. The algorithm works with small batches of samples, making predictions and updating the model parameters accordingly. Mini-batch gradient descent strikes a balance between the efficiency of batch gradient descent and the stochasticity of stochastic gradient descent.

By adjusting the batch size, we can control the trade-off between computational efficiency and the randomness of the learning process, enabling us to find an optimal balance for training our models effectively.

-   **Batch Gradient Descent**: `Batch Size = Size of Training Set`
-   **Stochastic Gradient Descent**: `Batch Size = 1`
-   **Mini-Batch Gradient Descent**: `1 < Batch Size < Size of Training Set`

In the case of mini-batch gradient descent, popular batch sizes include `32`, `64`, and `128` samples. You may see these values used in models in most tutorials.

**What if the dataset doesn't divide evenly by the batch size?**  
This can and _does_ happen often. It simply means that the final batch has fewer samples than the other ones. You can simply remove some samples from the dataset or change the batch size so that the number of samples in the dataset does divide evenly by the batch size. Most training scripts handle this automatically.  
这种情况可以而且确实经常发生。这仅仅意味着最后一批的样本比其他批次少。您只需从数据集中删除一些样本或更改批大小，以便数据集中的样本数均匀除以批大小。大多数训练脚本会自动处理此问题。

Now, let's discuss an epoch.  
现在，让我们讨论一个时代。

#### Epoch 时代[](https://rentry.org/llm-training#epoch "Permanent link")

The number of epochs is a hyperparameter that determines how many times the learning algorithm will iterate over the entire dataset.  
epoch 数是一个超参数，用于确定学习算法将遍历整个数据集的次数。

One (1) epoch signifies that each sample in the training dataset has been used once to update the model's internal parameters. It consists of one or more batches. For instance, if we have one batch per epoch, it corresponds to the batch gradient descent algorithm mentioned earlier.  
一 （1） 个 epoch 表示训练数据集中的每个样本都已使用一次来更新模型的内部参数。它由一个或多个批次组成。例如，如果我们每个 epoch 有一个批次，它对应于前面提到的批次梯度下降算法。

You can visualize the number of epochs as a "for-loop" iterating over the training dataset. Within this loop, there is another nested "for-loop" that goes through each batch of samples, where a batch contains the specified number of samples according to the batch size.  
您可以将 epoch 的数量可视化为遍历训练数据集的“for 循环”。在此循环中，还有另一个嵌套的“for-循环”，它遍历每批样本，其中批次包含根据批大小指定数量的样本。

To assess the model's performance over epochs, it's common to create line plots, also known as learning curves. These plots display epochs on the x-axis as time and the model's error or skill on the y-axis. Learning curves are useful for diagnosing whether the model has over-learned (high training error, low validation error), under-learned (low training and validation error), or achieved a suitable fit to the training dataset (low training error, reasonably low validation error). We will delve into learning curves in the next part.  
为了评估模型在各个时期的性能，通常创建折线图，也称为学习曲线。这些图在 x 轴上将纪元显示为时间，在 y 轴上显示模型的误差或技能。学习曲线可用于诊断模型是否过度学习（高训练误差、低验证误差）、学习不足（低训练和验证误差）或达到适合训练数据集的拟合（低训练误差、合理低的验证误差）。我们将在下一部分深入研究学习曲线。

Or do you still not understand the difference? In that case, let's look at the main difference between batches and epochs...  
还是您仍然不明白其中的区别？在这种情况下，让我们看看批次和纪元之间的主要区别......

#### Batch vs Epoch 批处理与纪元[](https://rentry.org/llm-training#batch-vs-epoch "Permanent link")

The batch size is a number of samples processed before the [model is updated](https://files.catbox.moe/0tboxf.png).  
批量大小是在更新模型之前处理的样本数。

The number of epochs is the number of complete passes through the training dataset.  
epoch 数是通过训练数据集的完整传递次数。

The size of a batch must be more than or equal to one (bsz=>1) and less than or equal equal to the number of samples in the training dataset (bsz=< No. Samples).  
批次的大小必须大于或等于 1 （bsz=>1） 且小于或等于训练数据集中的样本数 （bsz=< 否。示例）。

The number of epochs can be set to an **integer** values between one (1) and infinity. You can run the algorithm for as long as you like and even stop it using other criteria beside a fixed number of epochs, such as a change (or lack of change) in model error over time.  
纪元数可以设置为介于一 （1） 和无穷大之间的整数值。您可以根据需要运行该算法，甚至可以使用固定周期数之外的其他条件停止该算法，例如模型误差随时间的变化（或缺乏变化）。

They're both **integer** values and they're both hparams for the learning algorithm, e.g. parameters for the learning process, not internal model parameters found by the learning process.  
它们都是整数值，并且都是学习算法的参数，例如学习过程的参数，而不是学习过程找到的内部模型参数。

You must specify the batch size and number of epochs for a learning algorithm.  
您必须为学习算法指定批处理大小和纪元数。

There's no magic rule of thumb on how to configure these hparams. You should try and find the sweet spot for your specific use-case. :D  
关于如何配置这些 hparams，没有神奇的经验法则。您应该尝试为您的特定用例找到最佳位置。:D

**Here's a quick working example:  
下面是一个快速工作示例：**

Assume you have a dataset with 200 samples (rows or sequences of data) and you choose a batch size of 5 and 1,000 epochs. This means that the dataset will be divided into 40 batches, each with five samples. The model weights will be updated after each batch of five samples. This will also mean that one epoch will involve 40 batches or 40 updates to the model.  
假设您有一个包含 200 个样本（数据行或数据序列）的数据集，并且您选择的批处理大小为 5 和 1,000 个 epoch。这意味着数据集将分为 40 个批次，每个批次有 5 个样本。模型权重将在每批五个样品后更新。这也意味着一个纪元将涉及 40 个批次或 40 次模型更新。

With 1,000 epochs, the model will be exposed to the entire dataset 1,000 times. That is a total of 40,000 batches during the entire training process.  
使用 1,000 个 epoch，模型将暴露在整个数据集中 1,000 次。也就是说，在整个训练过程中总共有 40,000 个批次。

**Keep in mind that larger batch sizes result in higher GPU memory usage. We will be using Gradient Accumulation Steps to overcome this!  
请记住，批处理大小越大，GPU 内存使用率就越高。我们将使用梯度累积步骤来克服这个问题！**

### Learning Rate 学习率[](https://rentry.org/llm-training#learning-rate "Permanent link")

As discussed in the section for Batch and Epoch, in machine learning, we often use an optimization method called stochastic gradient descent (SGD) to train our models. One important factor in SGD is the learning rate, which determines how much the model should change in response to the estimated error during each update of its weights.  
正如 Batch 和 Epoch 部分所讨论的，在机器学习中，我们经常使用一种称为随机梯度下降 （SGD） 的优化方法来训练我们的模型。SGD 的一个重要因素是学习率，它决定了模型在每次更新其权重期间应根据估计误差的变化程度。

Think of the learning rate as a knob that controls the size of steps taken to improve the model. If the learning rate is too small, the model may take a long time to learn or get stuck in a suboptimal solution. On the other hand, if the learning rate is too large, the model may learn too quickly and end up with unstable or less accurate results.

Choosing the right learning rate is crucial for successful training. It's like finding the Goldilocks zone—not too small, not too large, but just right. You need to experiment and investigate how different learning rates affect your model's performance. By doing so, you can develop an intuition about how the learning rate influences the behavior of your model during training.

So, when fine-tuning your training process, pay close attention to the learning rate as it plays a significant role in determining how effectively your model learns and performs.

#### Learning Rate and Gradient Descent[](https://rentry.org/llm-training#learning-rate-and-gradient-descent "Permanent link")

Stochastic gradient descent estimates the error gradient for the current state of the model using examples from the training dataset, then updates the weights of the model using the [backpropagation of errors algorithm](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/), referred to as simply "backpropagation." The amount that the weights are updated during training is referred to as the step size or the "learning rate."

Specifically, the learning rate is a configurable hyperparameter used in training that has a very small positive value, often in the range between 0.0 and 1.0. (Note: between these values, not these values themselves.)

The learning rate controls how quickly the model is adapted to the problem. Smaller learning rate would require you to have more training **epochs**, since smaller changes are made to the weights with each update. Larger learning rates result in rapid changes and require fewer training epochs.

> high learning rate = less epochs.  
> low learning rate = more epochs.

"**The learning rate is perhaps the most important hyperparameter. If you have time to tune only one hyperparameter, tune the learning rate.**" —[Deep Learning](https://amzn.to/2NJW3gE)  
“学习率可能是最重要的超参数。如果您只有时间调整一个超参数，请调整学习率。

Let's learn how to configure the learning rate now.  
现在让我们学习如何配置学习率。

#### Configuring the Learning Rate  
配置学习率[](https://rentry.org/llm-training#configuring-the-learning-rate "Permanent link")

-   Start with a reasonable range: Begin by considering a range of learning rate values commonly used in similar tasks. Find out the learning rate used for the pre-trained model you're fine-tuning and base it off of that. For example, a common starting point is 1e-5 (0.00001), which is often found to be effective for transformer models.  
    从合理的范围开始：首先考虑类似任务中常用的学习率值范围。找出用于您正在微调的预训练模型的学习率，并以此为基础。例如，一个常见的起点是 1e-5 （0.00001），这通常被发现对变压器模型有效。
-   Observe the training progress: Run the training process with the chosen learning rate and monitor the model's performance during training. Keep an eye on metrics like loss or accuracy to assess how well the model is learning.  
    观察训练进度：使用所选的学习率运行训练过程，并在训练期间监视模型的性能。密切关注损失或准确性等指标，以评估模型的学习情况。
-   Too slow? If the learning rate is too small, you may notice that the model's training progress is slow, and it takes a long time to converge or make noticeable improvements. In cases like this, consider increasing the learning rate to speed up the learning process.  
    太慢了？如果学习率太小，您可能会注意到模型的训练进度很慢，并且需要很长时间才能收敛或做出明显的改进。在这种情况下，请考虑提高学习率以加快学习过程。
-   Too fast? If the learning rate is too large, the model may learn too quickly, leading to unstable results. Signs of a too high `lr` include wild fluctuations in loss or accuracy during training. If you observe this behaviour, consider decreasing the `lr`.  
    太快了？如果学习率过大，模型可能会学习得太快，导致结果不稳定。过高 `lr` 的迹象包括训练期间损失或准确性的剧烈波动。如果观察到此行为，请考虑减少 `lr` .
-   Iteratively adjust the learning rate: Based on the observations in steps 3 and 4, iteratively adjust the learning rate and re-run the training process. Gradually narrow down the range of learning rates that produce the best performance.  
    迭代调整学习率：根据步骤 3 和 4 中的观察结果，迭代调整学习率并重新运行训练过程。逐渐缩小产生最佳性能的学习率范围。

A general-purpose formula for calculating the learning rate is:  
计算学习率的通用公式为：

`base_lr * sqrt(supervised_tokens_in_batch / pretrained_bsz)`

The `base_lr` refers to the pre-trained model's learning rate. In case of Mistral, this is 5e-5. `supervised_tokens_in_batch` refers the total number of supervised tokens (axolotl reports this number once you start training), dividing that by total number of steps (also reported by axolotl) divided by the total number of epochs; i.e. `total_supervised_tokens / (num_steps / num_epochs)`. The `pretrained_bsz` refers to the original batch size of the base model. In case of Mistral and Llama, this is 4,000,000 (4 millions). For example, let's assume we are training Mistral with a dataset that contains 2 million supervised tokens, and we're training on a single GPU at batch size 1. Let's also assume this takes 350 steps. The final formula would look like this:  
是指 `base_lr` 预训练模型的学习率。如果是米斯特拉尔，这是 5e-5。 `supervised_tokens_in_batch` 指受监督令牌的总数（一旦您开始训练，蝾螈就会报告这个数字），将其除以总步数（也由蝾螈报告）除以总周期数;即 `total_supervised_tokens / (num_steps / num_epochs)` .是指 `pretrained_bsz` 基本模型的原始批量大小。在米斯特拉尔和骆驼的情况下，这是 4,000,000（4 百万）。例如，假设我们正在使用包含 200 万个监督令牌的数据集来训练 Mistral，并且我们正在以批处理大小 1 在单个 GPU 上进行训练。我们还假设这需要 350 个步骤。最终公式如下所示：

`5e-5 * sqrt(2000000/(350/1) / 4000000) = 0.00000189` (1.89e-6)  `5e-5 * sqrt(2000000/(350/1) / 4000000) = 0.00000189` （1.89e-6）

For reference, the base learning rate for Llama-2 models is 3e-4.  
作为参考，Llama-2 模型的基本学习率为 3e-4。

### Gradient Accumulation 梯度累积[](https://rentry.org/llm-training#gradient-accumulation "Permanent link")

Higher batch sizes result in higher memory consumption. Gradient accumulation aims to fix this.  
批大小越大，内存消耗就越高。梯度累积旨在解决这个问题。

Gradient accumulation is a mechanism to split the batch of samples - used for training your model - into several mini-batches of samples that will be run sequentially.  
梯度累积是一种机制，用于将用于训练模型的样本批次拆分为多个小批量样本，这些样本将按顺序运行。

[![gradient.png](https://s8d2.turboimg.net/sp/1e30fd5830a99908a80d8c3950030a22/gradient.png "gradient.png")](https://www.turboimagehost.com/p/89525001/gradient.png.html) Source: [Towards Data Science](https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa)

First, let's see how backpropagation works.

#### Backpropagation[](https://rentry.org/llm-training#backpropagation "Permanent link")

In a model, we have many layers that work together to process our data. Think of these layers as interconnected building blocks. When we pass our data through the model, it goes through each of these layers, step by step, in a forward direction. As it travels through the layers, the model makes predictions for the data.

After the data has gone through all the layers and the model has made predictions, we need to measure how accurate or "right" the model's predictions are. We do this by calculating a value called the "loss." The loss tells us how much the model deviates from the correct answers for each data sample.

Now comes the interesting part. We want our model to learn from its mistakes and improve its predictions. To do this, we need to figure out how the loss value changes when we make small adjustments to the model's internal parameters, like the weights and biases.

This is where the concept of gradients comes in. Gradients help us understand how the loss value changes with respect to each model parameter. Think of gradients as arrows that show us the direction and magnitude of the change in the loss as we tweak the parameters.

Once we have the gradients, we can use them to update the model's parameters and make them better. We choose an optimizer, which is like a special algorithm responsible for guiding these parameter updates. The optimizer takes into account the gradients, as well as other factors like the learning rate (how big the updates should be) and momentums (which help with the speed and stability of learning).

To simplify, let's consider a popular optimization algorithm called stochastic gradient descent (SGD). It's like a formula: V = V - (lr \* grad). In this formula, V represents any parameter in the model that we want to update (like weights or biases), lr is the learning rate that controls the size of the updates, and grad is the gradients we calculated earlier. This formula tells us how to adjust the parameters based on the gradients and the learning rate.

In summary, backpropagation is a process where we calculate how wrong our model is by using the loss value. We then use gradients to understand which direction to adjust our model's parameters. Finally, we apply an optimization algorithm, like stochastic gradient descent, to make these adjustments and help our model learn and improve its predictions.

#### Gradient Accumulation explained[](https://rentry.org/llm-training#gradient-accumulation-explained "Permanent link")

Gradient accumulation is a technique where we perform multiple steps of computation without updating the model's variables. Instead, we keep track of the gradients obtained during these steps and use them to calculate the variable updates later. It's actually quite simple!  
梯度累积是一种技术，我们在不更新模型变量的情况下执行多个计算步骤。相反，我们会跟踪在这些步骤中获得的梯度，并在以后使用它们来计算变量更新。其实很简单！

To understand gradient accumulation, let's think about splitting a batch of samples into smaller groups called mini-batches. In each step, we process one of these mini-batches without updating the model's variables. This means that the model uses the same set of variables for all the mini-batches.  
为了理解梯度累积，让我们考虑将一批样本分成更小的组，称为小批量。在每个步骤中，我们都会处理其中一个小批量，而无需更新模型的变量。这意味着该模型对所有小批量使用相同的变量集。

By not updating the variables during these steps, we ensure that the gradients and updates calculated for each mini-batch are the same as if we were using the original full batch. In other words, we guarantee that the sum of the gradients obtained from all the mini-batches is equal to the gradients obtained from the full batch.  
通过在这些步骤中不更新变量，我们确保为每个小批量计算的梯度和更新与使用原始完整批次相同。换言之，我们保证从所有小批量中获得的梯度之和等于从整批中获得的梯度。

To summarize, gradient accumulation allows us to divide the batch into mini-batches, perform computation on each mini-batch without updating the variables, and then accumulate the gradients from all the mini-batches. This accumulation ensures that we obtain the same overall gradient as if we were using the full batch.  
总而言之，梯度累积允许我们将批次划分为小批量，在不更新变量的情况下对每个小批量执行计算，然后从所有小批量中累积梯度。这种累积确保了我们获得与使用完整批次相同的整体梯度。

#### Iteration 迭 代[](https://rentry.org/llm-training#iteration "Permanent link")

So, let's say we are accumulating gradients over 5 steps. In the first 4 steps, we don't update any variables, but we store the gradients. Then, in the fifth step, we combine the accumulated gradients from the previous steps with the gradients of the current step to calculate and assign the variable updates.  
因此，假设我们正在 5 个步骤中累积梯度。在前 4 个步骤中，我们不更新任何变量，但存储梯度。然后，在第五步中，我们将前面步骤的累积梯度与当前步骤的梯度相结合，以计算和分配变量更新。

During the first step, we process a mini-batch of samples. We go through the forward and backward pass, which allows us to compute gradients for each trainable model variable. However, instead of actually updating the variables, we focus on storing the gradients. To do this, we create additional variables for each trainable model variable to hold the accumulated gradients.  
在第一步中，我们处理一小批样品。我们通过前向和后向传递，这使我们能够计算每个可训练模型变量的梯度。但是，我们不是实际更新变量，而是专注于存储梯度。为此，我们为每个可训练的模型变量创建额外的变量来保存累积的梯度。

After computing the gradients for the first step, we store them in the respective variables we created for the accumulated gradients. This way, the gradients of the first step will be accessible for the following steps.  
在计算出第一步的梯度后，我们将它们存储在我们为累积梯度创建的相应变量中。这样，第一步的梯度将可用于以下步骤。

We repeat this process for the next three steps, accumulating the gradients without updating the variables. Finally, in the fifth step, we have the accumulated gradients from the previous four steps and the gradients of the current step. With these gradients combined, we can compute the variable updates and assign them accordingly. Here's an illustration:  
在接下来的三个步骤中，我们重复此过程，在不更新变量的情况下累积梯度。最后，在第五步中，我们得到了前四个步骤的累积梯度和当前步骤的梯度。将这些梯度组合在一起，我们可以计算变量更新并相应地分配它们。下面是一个插图：

[![The value of the accumulated gradients at the end of N steps](https://s8d3.turboimg.net/sp/d996675955a792ebc17ed07b1d7ae78b/Screenshot_from_2023-05-29_18-29-08.png "The value of the accumulated gradients at the end of N steps")](https://www.turboimagehost.com/p/89525118/Screenshot_from_2023-05-29_18-29-08.png.html)

Now the second step starts, and again, all the samples of the second mini-batch propagate through all the layers of the model, computing the gradients of the second step. Just like the step before, we don't want to update the variables yet, so there's no need in computing the variable updates. What's different than the first step though, is that instead of just storing the gradients of the second step in our variables, we're going to add them to the values stored in the variables, which currently hold the gradients of the first step.  
现在第二步开始，第二个小批量的所有样本再次传播到模型的所有层中，计算第二步的梯度。就像前面的步骤一样，我们还不想更新变量，因此无需计算变量更新。不过，与第一步不同的是，我们不仅将第二步的梯度存储在变量中，而是将它们添加到变量中存储的值中，这些变量当前包含第一步的梯度。

Steps 3 and 4 are pretty much the same as the second step, as we're not yet updating the variables, and we're accumulating the gradients by adding them to our variables.  
第 3 步和第 4 步与第二步几乎相同，因为我们尚未更新变量，而是通过将梯度添加到变量中来累积梯度。

Then in step 5, we do want to update the variables, as we intended to accumulate the gradients over 5 steps. After computing the gradients of the fifth step, we will add them to the accumulated gradients, resulting in the sum of all the gradients of those 5 steps. We'll then take this sum and insert it as a parameter to the optimizer, resulting in the updates computed using all the gradients of those 5 steps, computed over all the samples in the global batch.  
然后在第 5 步中，我们确实希望更新变量，因为我们打算在 5 个步骤中累积梯度。在计算了第五步的梯度后，我们将把它们添加到累积的梯度中，得到这 5 个步骤的所有梯度的总和。然后，我们将获取此总和并将其作为参数插入优化器，从而使用这 5 个步骤的所有梯度计算更新，这些梯度是在全局批处理中的所有样本上计算的。

If we use SGD as an example, let's se the variables after the updates at the end of the fifth step, computed using the gradients of those 5 steps (N=5 in the following example):  
如果我们以 SGD 为例，让我们在第五步结束时更新后的变量，使用这 5 个步骤的梯度（在以下示例中为 N=5）进行计算：  
[![The value of a trainable variable after N steps (using SGD)](https://s8d3.turboimg.net/sp/f65dda03b8b23fb5af2582c963a571b5/Screenshot_from_2023-05-29_18-34-13.png "The value of a trainable variable after N steps (using SGD)")](https://www.turboimagehost.com/p/89525120/Screenshot_from_2023-05-29_18-34-13.png.html)

#### Configuring the number of gradient accumulation steps[](https://rentry.org/llm-training#configuring-the-number-of-gradient-accumulation-steps "Permanent link")

As we extensively discussed, you'll want to use gradient accumulation steps to achieve an effective batch size that is close to or larger than the desired batch size.

For example, if your desired batch size is 32 samples but you have limited VRAM that can only handle a batch size of 8, you can set the gradient accumulation steps to 4. This means that you accumulate gradients over 4 steps before performing the update, effectively simulating a batch size of 32 (8 \* 4).  
例如，如果所需的批量大小为 32 个样本，但 VRAM 有限，只能处理 8 个样本，则可以将梯度累积步长设置为 4。这意味着在执行更新之前，您可以在 4 个步骤中累积梯度，从而有效地模拟 32 （8 \* 4） 的批处理大小。

In general, I'd recommend balancing the gradient accumulation steps with the available resources to maximize your computational efficiency. Too few accumulation steps may result in insufficient gradient information, while too many would increase memory requirements and slow down the training process.  
一般来说，我建议在梯度累积步骤与可用资源之间取得平衡，以最大限度地提高计算效率。累积步骤太少可能会导致梯度信息不足，而累积步骤过多会增加内存需求并减慢训练过程。

This section is being worked on right now.  
本节目前正在处理中。

___

## Interpreting the Learning Curves  
解释学习曲线[](https://rentry.org/llm-training#interpreting-the-learning-curves "Permanent link")

Learning curves are one of the most common tools for algorithms that learn incrementally from a training dataset. The model will be evaluated using a validation split, and a plot is created for the loss function, measuring how different the model's current output is compared to the expected one. Let's try and go over the specifics of learning curves, and how they can be used to diagnose the learning and generalization behaviour of your model.  
学习曲线是从训练数据集中增量学习的算法的最常用工具之一。将使用验证拆分对模型进行评估，并为损失函数创建一个图，以测量模型的当前输出与预期输出相比的差异。让我们尝试回顾一下学习曲线的细节，以及如何使用它们来诊断模型的学习和泛化行为。

### Overview 概述[](https://rentry.org/llm-training#overview "Permanent link")

A learning curve can be likened to a graph that presents the relationship between time or experience (x-axis) and the progress or improvement in learning (y-axis), using a more technical explanation.  
学习曲线可以比作一个图表，它使用更技术性的解释来呈现时间或经验（x轴）与学习进度或改进（y轴）之间的关系。

Let's take the example of learning the Japanese language. Imagine that you're embarking on a journey to learn Japanese, and every week for a year, you evaluate your language skills and assign a numerical score to measure your progress. By plotting these scores over the span of 52 weeks, you can create a learning curve that visually illustrates how your understanding of the language has evolved over time.  
让我们以学习日语为例。想象一下，您正在踏上学习日语的旅程，在一年的时间里，您每周都会评估自己的语言技能并分配一个数字分数来衡量您的进步。通过绘制 52 周内的这些分数，您可以创建一个学习曲线，直观地说明您对语言的理解如何随着时间的推移而演变。

> **Line plot of learning (y-axis) over experience (x-axis).  
> 学习（y 轴）与经验（x 轴）的折线图。**

To make it more meaningful, let's consider a scoring system where lower scores represent better learning outcomes. For instance, if you initially struggle with basic vocabulary and grammar, your scores may be higher. However, as you continue learning and make progress, your scores will decrease, indicating a more solid grasp of the language. Ultimately, if you achieve a score of 0.0, it would mean that you have mastered Japanese perfectly, without making any mistakes during your learning journey.  
为了使其更有意义，让我们考虑一个评分系统，其中较低的分数代表更好的学习成果。例如，如果您最初在基本词汇和语法方面遇到困难，您的分数可能会更高。但是，随着您继续学习并取得进步，您的分数会下降，表明对语言的掌握更加扎实。最终，如果您达到 0.0 分，则意味着您已经完美地掌握了日语，在学习过程中没有犯任何错误。

___

During the training process of a model, we can assess its performance at each step. This assessment can be done on the training dataset to see how well the model is **_learning_**. Additionally, we can evaluate it on a separate validation dataset that was not used for training to understand how well the model is able to **_generalize._**  
在模型的训练过程中，我们可以评估其每一步的性能。可以在训练数据集上进行此评估，以查看模型的学习情况。此外，我们可以在未用于训练的单独验证数据集上对其进行评估，以了解模型的泛化能力。

Here are two types of learning curves that are commonly used:  
以下是常用的两种学习曲线：

-   Train Learning Curve: This curve is derived from the training dataset and gives us an idea of how well the model is learning during training.  
    训练学习曲线：该曲线源自训练数据集，让我们了解模型在训练期间的学习情况。
-   Validation Learning Curve: This curve is created using a separate validation dataset. It helps us gauge how well the model is generalizing to new data.  
    验证学习曲线：此曲线是使用单独的验证数据集创建的。它帮助我们衡量模型对新数据的泛化程度。

It's often beneficial to have dual learning curves for both the train and validation datasets.  
对训练数据集和验证数据集进行对偶学习通常是有益的。

Sometimes, we might want to track multiple metrics for evaluation. For example, in classification problems, we might optimize the model based on cross-entropy loss and evaluate its performance using classification accuracy. In such cases, we create two plots: one for the learning curves of each metric. Each plot can show two learning curves, one for the train dataset and one for the validation dataset.  
有时，我们可能希望跟踪多个指标进行评估。例如，在分类问题中，我们可以基于交叉熵损失优化模型，并使用分类精度评估其性能。在这种情况下，我们创建了两个图：一个用于每个指标的学习曲线。每个图可以显示两条学习曲线，一条用于训练数据集，另一条用于验证数据集。

We refer to these types of learning curves as:  
我们将这些类型的学习曲线称为：

-   Optimization Learning Curves: These curves are calculated based on the metric (e.g., loss) used to optimize the model's parameters.  
    优化学习曲线：这些曲线是根据用于优化模型参数的指标（例如损失）计算得出的。
-   Performance Learning Curves: These curves are derived from the metric (e.g., accuracy) used to evaluate and select the model.  
    性能学习曲线：这些曲线源自用于评估和选择模型的指标（例如准确性）。

By analyzing these learning curves, we gain valuable insights into the model's progress and its ability to learn and generalize effectively.  
通过分析这些学习曲线，我们可以获得对模型进展及其有效学习和泛化能力的宝贵见解。

Now that you know a bit more about learning curves, let's take a look at some common shapes observed in learning curve plots.  
现在您对学习曲线有了更多的了解，让我们来看看在学习曲线图中观察到的一些常见形状。

### Model Behaviour Diagnostics  
模型行为诊断[](https://rentry.org/llm-training#model-behaviour-diagnostics "Permanent link")

The shape and dynamics of a learning curve can be used to diagnose the behaviour of a model and in turn perhaps suggest at the type of configuration changes that may be made to improve learning an/or performance.  
学习曲线的形状和动态可用于诊断模型的行为，进而可能建议为提高学习和/或性能而进行的配置更改类型。

There are three (3) common dynamics that you're likely to observe in learning curves:  
您可能会在学习曲线中观察到三 （3） 种常见动态：

-   Underfit. 欠拟合。
-   Overfit. 过拟合。
-   Well-fit. 合身。

We'll take a closer look at each example. The examples will assume you're looking at a **minimizing** metric, meaning that **smaller** relative scores on the y-axis indicate **better** learning.  
我们将仔细研究每个示例。这些示例将假设您正在查看最小化指标，这意味着 y 轴上的相对分数越小，表示学习效果越好。

#### Underfit Learning Curves 欠拟合学习曲线[](https://rentry.org/llm-training#underfit-learning-curves "Permanent link")

Refers to a model that cannot learn the training dataset. You can easily identify an underfit model from the curve of the training loss only.  
指无法学习训练数据集的模型。您仅可以从训练损失曲线中轻松识别欠拟合模型。

It may show a flatline or noisy values of relatively high loss, indicating that the model was unable to learn the training dataset at all. Let's take a look at the example below, which is common when the model doesn't have a suitable capacity for the complexity of the dataset:  
它可能显示相对高损失的平线或噪声值，表明模型根本无法学习训练数据集。让我们看一下下面的示例，当模型没有适合数据集复杂性的容量时，这很常见：

![An underfit model](https://s8d3.turboimg.net/sp/c391530b5442690ae2bf8f54396bc635/underfit.png "An underfit model")  
Underfit model. Source: [Machine Learning Mastery](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)  
欠拟合模型。来源：机器学习精通

An underfit plot is characterized by:  
欠拟合图的特征是：

-   The training loss remaining flat regardless of training.  
    无论训练如何，训练损失都保持不变。
-   The training loss continues to decrease until the end of training.  
    训练损失持续减少，直到训练结束。

#### Overfit Learning Curves 过拟合学习曲线[](https://rentry.org/llm-training#overfit-learning-curves "Permanent link")

This would refer to a model that has learned the training dataset **too well**, leading to a memorization of the data rather than generalization. This would include the statistical noise or random fluctuations present in the training dataset.  
这是指一个模型对训练数据集的学习能力太强，导致数据的记忆而不是泛化。这将包括训练数据集中存在的统计噪声或随机波动。

The problem with overfitting is that the more specialized the model becomes to the training data, the less well it's able to generalize to new data, resulting in an increase in generalization error. This increase in generalization error can be measured by the performance of the model on the validation dataset. This often happens if the model has more capacity than is necessary for the required problem, and in turn, too much flexibility. It can also occur if the model is trained for too long.  
过度拟合的问题在于，模型对训练数据的专业化程度越高，它对新数据的泛化能力就越差，从而导致泛化误差增加。泛化误差的增加可以通过模型在验证数据集上的性能来衡量。如果模型的容量超过所需问题所需的容量，则通常会发生这种情况，进而导致灵活性过大。如果模型训练时间过长，也会发生这种情况。

A plot of learning curves show overfitting if:  
如果出现以下情况，学习曲线图显示过拟合：

-   The plot of training loss continues to decrease with experience.  
    随着经验的增加，训练损失的图继续减少。
-   The plot of validation loss decreases to a point and begins increasing again.  
    验证损失图减小到一个点，然后再次开始增加。

The inflection point in validation loss may be the point at which training could be halted as experience after the point shows the dynamics of overfitting. Here's an example plot for an overfit model:  
验证损失的拐点可能是训练可以停止的拐点，因为在该点显示过拟合的动态之后的经验。下面是过拟合模型的示例图：

[![An overfit model](https://s8d7.turboimg.net/sp/ad8e283bd9f772d5290f5a6902f51d37/overfit.png "An overfit model")](https://www.turboimagehost.com/p/89521651/overfit.png.html)

#### Well-fit Learning Curves 拟合良好的学习曲线[](https://rentry.org/llm-training#well-fit-learning-curves "Permanent link")

This would be your goal during training - a curve between an overfit and underfit model.  
这将是您在训练期间的目标 - 过拟合和欠拟合模型之间的曲线。

A good fit is usually identified be a training and validation loss that decrease to a point of stability with a minimal gap between the two final loss values.  
良好的拟合通常被确定为训练和验证损失，该损失降低到稳定点，两个最终损失值之间的差距最小。

The loss of the model will always be lower on the training dataset than the validation dataset. This means that we should expect some gap between the train and validation loss learning curves. This gap is referred to as the "generalization gap."  
训练数据集上的模型损失始终低于验证数据集。这意味着我们应该预料到训练和验证损失学习曲线之间存在一些差距。这种差距被称为“泛化差距”。

This example plot would show a well-fit model:  
此示例图将显示一个拟合良好的模型：  
[![A well-fit model](https://s8d5.turboimg.net/sp/eeee5703561a4b5862e28eb01510cc34/wellfit.png "A well-fit model")](https://www.turboimagehost.com/p/89521681/wellfit.png.html)