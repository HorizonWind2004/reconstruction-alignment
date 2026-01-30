# 复现 RecA 的关键细节

有不少人想在自己的/其他 UMM 架构上复现 RecA，却常常忽略一些关键细节，导致性能不尽如人意。

我们认为 RecA 的核心思想是：利用来自图片的 semantic feature 去指导图片生成，也就是 **semantic-to-pixel generation 提升 text-to-pixel generation**。因此，引入 **information bottleneck** 是必要的，否则模型容易走一个 shortcut，**退化为纯粹专注于重建的 auto-encoder**，最终导致模式崩塌。那么，有哪些需要注意的点呢？

## 360 Prompt Template

我们提供了 360 条用于重建的 prompts，你可以在[这里](./BAGEL/data/consts.py)找到它们。

对于 Show-o 这类架构，我们发现使用多样化的 prompts，相比仅使用 "Describe the image" 一条 prompt 效果好不少。而对于 Harmon 使用 360 prompts 的效果差异不大，所以这条因模型而异。不过使用 360 prompts 总不会错。

## Resize 输入图像到最小分辨率

你需要将输入图像缩小到到模型在 **image-to-text training** 阶段所使用的 **最小可接受分辨率**。例如：

- **BAGEL**: 224×224
- **Show-o (VQGAN variant)**: 256×256

为什么要这样做？有不少文章发现更高分辨率的 visual understanding embeddings 会保留更多的像素细节。为了鼓励模型关注 semantic-level 的重建，而非 pixel-level 的复制，我们将输入图像缩放到 UMM 可接受的最小分辨率。这有助于模型在 RecA 训练过程中学习更抽象的 semantic representations。我们论文中的 ablation study 也验证了这一点：


<div align="center">
  <img src="assets/resolution.png" alt="Resolution Ablation Study" width="60%">
</div>

更有意思的是：如果输入图像的分辨率与生成相同，且它们处于**统一的表示空间**（举个例子，都是 VQGAN token，或者像 RAE 一样都是 siglip feature），模型很容易学会直接 copy-and-paste，从而导致模式崩塌。以 Show-o 的 VQGAN 版为例，如果我们输入 512x512 的图像（对应 16x16 个 VQ tokens），并让它重建 16x16 个 VQ tokens，模型的内部表示空间就会崩溃，训练几千步后 CE loss 就降为 0 了。将输入图像缩放到 256x256 就可解决（首选方案）。或者对输入图像进行模糊处理（次选方案）。

<div align="center">
  <img src="Show-o/assets/VQGAN.png" alt="" width="60%">
</div>

Janus 的输入输出分辨率虽然都是 384x384，但输入是 SigLIP features，输出是 VQGAN features，二者之间天然存在 information bottleneck，因此无需特殊处理。

## 尽可能冻结理解组件

对于生成与理解**解耦**的模型，应尽可能冻结理解组件。比如 BAGEL 和 Metaquery 类架构。

对于参数共享的模型（如 Janus、Show-o、Harmon），我们无法冻结理解部分，因此必须加入 image-to-text 数据以保持模型的理解能力。否则，模型的内部表示空间可能崩溃，生成能力也可能下降。
