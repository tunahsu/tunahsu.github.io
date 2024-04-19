---
title: "RepLKNet"
slug: replknet
date: 2022-09-07T16:05:31+08:00
categories:
- deep learning
- cnn
tags:
- replknet
- large kernel
thumbnailImage: https://i.imgur.com/9TIH6oH.png
---

[Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs](https://arxiv.org/abs/2203.06717) 近年來 Transformer 的崛起，普遍認為 self-attention 在影像領域可以表現得比 CNN 更好，這篇發表在 CVPR 2022 上的研究表示認為這不是因為 Self-attention 的設計形式(query-key-value)，而是因為其有效感受野特別大，因此作者提出了提出了超大 kernel 的模型，在一系列的實驗下證明較大的卷積核在現代模型優化的設計下，計算量並不會提升多少且在一些 downstream tasks 的效能更甚於較深但 kernel 較小的網路架構。

<!--more-->

# 貢獻

- 證明超大卷積核在過去沒人用，不代表現在不能用，在現代 CNN 設計的加持下， kernel 越大效能可能越好。
- 發現超大 depthwise 卷積並不會增加多少 FLOPs，如果再加上底層優化速度會更快，31x31 的計算密度最高可達 3x3 的 70 倍。
- 大卷積核不只能用在大的 feature map 上，作者發現在 7x7 的 feature map 上用 13x13 的卷積核都能漲點。
- ImageNet 的準確率並不能說明一切，作者發現在一些 downstream tasks(object detection、semantic segmentation...) 上的性能可能跟 ImageNet 的關係不大。
- 超深的 CNN 由大量的 3x3 kernel 堆疊而成，所以感受野很大，其實不是這樣，作者發現反而是少量的大 kernel 有效感受野必較大。

# 實驗

- 通過一系列的實驗，作者總結了在 CNN 中應用超大卷積核的五個準則:
    + 使用 Depthwise convolution
    + 添加 Shortcuts
    + 做結構重參數化(參考: https://zhuanlan.zhihu.com/p/361090497)
    + 要看在 downstream tasks 上的性能，不能只看 ImageNet
    + 小的 feature map 上也可以用大的卷積核

<figure align="center">
    <img src="https://i.imgur.com/wmQrin1.png">
    <figcaption>RepLKNet 架構</figcaption>
</figure>

- 基於以上準則，借鑑 Swim Transformer 的架構，提出新的架構 RepLKNet，其中使用大量的超大卷積核，如 27x27、31x31等。此架構非常簡單，其餘部分都是 1x1 卷積、batch norm 且完全沒有任何 attention。
- 基於超大卷積核，對有效感受野、shape bias(model 做決定時是看 feature 的形狀還是局部紋理)、Transformers 之所以性能好的原因等等進行討論及分析，作者發現 ResNet-152 等傳統深層小 kernel 模型的有效感受野其實不大，反而是大 kernel 模型的有效感受野較大且更接近人類視覺(shape bias 高)。Transformer 的關鍵能在於較大的有效感受野而不是 self-attention 的設計形式。

<figure align="center">
    <img src="https://i.imgur.com/R2G2YS8.png">
    <figcaption>有效感受野視覺化</figcaption>
</figure>

- 作者在 ImageNet、Cityscapes、ADE20K、COCO 等資料集上進行測試，皆取得不錯的成果，由於篇幅較長這邊省略不談，詳見原論文。

# 結論

作者視覺化了 RepLKNet-31、RepLKNet-13、ResNet-101、ResNet-152 的有效感受野(方法參考原論文)，發現大 kernel 的模型有效感受野遠超深層小 kernel 模型。並且研究了模型的 shape bias，人類的 shape bias 約為 90% 左右，如下圖左邊的菱形點，其中比較的模型包含 Swin、ResNet152、RepLKNet-31、RepLKNet-3，發現 RepLKNet-3 和 ResNet-152 的 kernel size 一樣大，shape bias 也較為接近，相關研究發現其實 Swim(局部 attention)的 shape bias 並不高，而 ViT(全局 attention)的卻很高，這似乎說明 attention 的形式並不是關鍵，作用的範圍才是關鍵，這也解釋了為甚麼 RepLKNet-31 具有較高的 shape bias。

<figure align="center">
    <img src="https://i.imgur.com/wWM2bD4.png">
    <figcaption>Shape bias 比較</figcaption>
</figure>