---
title: "GaborNet 在卷積神經網路中可學習的 Gabor 濾波器參數"
slug: gabornet
date: 2022-05-02T22:56:56+08:00
categories:
- deep learning
- cnn
tags:
- gabornet
- gabor filter
thumbnailImage: https://i.imgur.com/bVLduli.png
---

以 Dennis Gabor 命名的 Gabor 濾波器，是一種用於紋理分析的線性濾波器，主要分析的是影像在特定區域的特定方向上是否有特定頻率的內容，許多視覺科學家認為 Gabor 的頻率和方向的表達與人類的視覺系統很類似。研究發現，Gabor 濾波器特別適合用於紋理表示和辨識。

<!--more-->

# Abstract

[GaborNet: Gabor filters with learnable parameters in deep convolutional neural network](https://ieeexplore.ieee.org/abstract/document/9030571) 中針對深度卷積網路的影像辨識系統，提出了一種基於 Gabor 濾波器的卷積層，主要於收斂訴的提升與訓練複雜度上做出改善，其中 Gabor function 中的參數是可以透過反向傳播來更新的。該系統主要是用 Python 來實現，他們在幾個 dataset 上進行測試，其結果優於傳統的卷積神經網路

# Theory

傅立葉轉換在訊號處理上可以幫助我們將數位影像從空間域轉到頻率域，擷取到在空間域上不容易取得的特徵，但是經過傅立葉轉換後影像在不同位置頻率特徵往往會混和在一起，但是 Gabor 濾波器卻可以擷取局部空間的頻率特徵，是一個很好的紋理檢測工具

在二維空間中將一個**三角函數**(本文使用餘弦函數)與一個**高斯函數**疊加我們就可以得到 Gabor 濾波器，如下圖

<figure align="center">
    <img src="https://i.imgur.com/LasA4c1.png">
    <figcaption>2D Gabor function</figcaption>
</figure>

原文使用餘弦函數與高斯函數相乘來得到 Gabor function，但文中並未對其中的各項參數做詳細的講解，以下將會逐一介紹

$$ g(x, y, \omega, \theta, \varphi, \sigma) = exp(-\frac{x ^ {’2} + y ^ {’2}}{2 \sigma ^ 2}) cos(\omega x ^ {’} + \varphi) $$

- Orientation $ \theta $：控制 Gabor 濾波器中條帶的方向，有效值為 0 ~ 360 度的實數
- Phase offset $ \phi $：表示餘弦函數的相位偏移參數，有效值為 0 ~ 180 度的實數
- Frequency $ \omega $：餘弦函數中的頻率參數，頻率越高，黑白相間的間隔越小
- Standard deviation $ \sigma $：表示高斯函數中的標準差，該參數決定了 Gabor 濾波器都可接受區域大小，其值可設為 $ \pi / \omega $

在下圖中顯示了 Alexnet 中第一層卷積層中使用的 96 個濾波器，從紋理上來看幾乎與 Gabor 濾波器一樣，由此可知我們在卷積神經網路中使用 Gabor 濾波器是可行的

<figure align="center">
    <img src="https://i.imgur.com/mRtrTqc.png">
    <figcaption>Filters of fisrt layer in Alexnet</figcaption>
</figure>

# Experiment

實驗的部分作者分別在 Dogs vs Cats/AffectNet/ImageNet 三個資料集上進行測試，針對前兩個資料集設計了兩個簡單的傳統 CNN 模型，並將模型的第一層卷積層替換成 Gabor Layer 來與原本的架構做比較，而在 ImageNet dataset 上則是使用 Alexnet 來測試，詳細的架構圖可參考[原文](https://ieeexplore.ieee.org/abstract/document/9030571)，其結果由下圖可知看出 Gabor CNN 在訓練的收斂速度與辨識準確度都要優於傳統的 CNN

<figure align="center">
    <img src="https://i.imgur.com/ATWS2XP.png">
    <figcaption>Performance on Dogs vs Cats</figcaption>
</figure>

# Discussion

在實驗結果中顯示，有 Gabor Layer 的 CNN 明顯有更好的性能，尤其是在貓與狗的分類上準確度更是提升了 6 個百分點，並且參數量明顯減少了許多。但是如果資料集中只有少部分的圖片含有 Gabor 特徵，如 ImageNet，那這種方法將不會提高辨識的準確度，不過 Gabor CNN 仍然能夠在相同的準確度下減少參數量且提升訓練速度

# Demo

最後為了驗證此篇論文的方法，使用了作者在 GitHub 提供的 GaborConv2d API 來設計一個簡單的 Gabor CNN 模型，與原文的方法一樣用 Gabor Layer 替換了傳統 CNN 模型中的第一層卷積層來訓練 Dogs vs Cats 資料集，不過模型架構並未完全參照原文，而是使用了四層簡單的 CONV >> ReLU >> Max Pool 來做訓練，模型定義如下

```
class GaborCNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # ReLU
        self.relu = nn.ReLU(inplace=True)
        # Max pool
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Convolution 1, input shape = (3, 224, 224)
        self.conv1 = GaborConv2d(in_channels=3, out_channels=16, kernel_size=(13, 13), device='cuda')
        # Convolution 2, input shape = (16, 106, 106)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        # Convolution 3, input shape = (32, 52, 52)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # Convolution 4, input shape = (64, 25, 25)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        # Fully connected, input shape = (128 * 11 * 11)
        self.fc1 = nn.Linear(128 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.maxpool(self.relu(self.conv3(x)))
        x = self.maxpool(self.relu(self.conv4(x)))
        x = x.view(-1, 128 * 11 * 11)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
```

由於運算設備的限制，這裡我只從原資料集的訓練集(25000張)中取了 5000 + 5000 張貓狗各半來切成訓練集驗證集，分別為 80% 與 20%，訓練結果如下表格所示，雖然在準確度上並無提升，但是一般 CNN 與 Gabor CNN 的參數量分別為 2457890 與 2088146，後者少了約 1/6 的參數量，由此可知其性能的確是優於一般的 CNN

<br>
<table style="text-align:center">
    <tr>
        <td>Accuracy</td>
        <td>Train</td>
        <td>Train</td>
        <td>Val</td>
        <td>Val</td>
    </tr>
    <tr>
        <td>Epoch</td>
        <td>CNN</td>
        <td>Gabor CNN</td>
        <td>CNN</td>
        <td>Gabor CNN</td>
    </tr>
    <tr>
        <td>1</td>
        <td>0.611</td>
        <td>0.581</td>
        <td>0.659</td>
        <td>0.632</td>
    </tr>
    <tr>
        <td>10</td>
        <td>0.812</td>
        <td>0.794</td>
        <td>0.803</td>
        <td>0.782</td>
    </tr>
    <tr>
        <td>20</td>
        <td>0.860</td>
        <td>0.843</td>
        <td>0.845</td>
        <td>0.815</td>
    </tr>
    <tr>
        <td>30</td>
        <td>0.886</td>
        <td>0.870</td>
        <td>0.862</td>
        <td>0.864</td>
    </tr>
    <tr>
        <td>40</td>
        <td>0.901</td>
        <td>0.894</td>
        <td>0.868</td>
        <td>0.861</td>
    </tr>
</table>


