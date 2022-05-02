---
title: "DynCNN 用於監控影像的動態卷積神經網路"
slug: dyncnn
date: 2022-04-13T15:16:10+08:00
categories:
- deep learning
- cnn
tags:
- dyncnn
- surveillance
thumbnailImage: https://i.imgur.com/x6589qS.png
---

智慧監控所使用的CNN的架構大部分是來自 ImageNet Challenge 比賽中獲勝的網路架構，這些較著名的CNN 架構具有更深層且更複雜的神經網路從而達到更高的精度，但在現今的硬體技術發展下，高端硬體設備已經可以讓這些複雜的神經網路達成 real-time 的效果。但在智慧監控領域中多通道的影像，需要同時進行處理並實現及時運算，考量這些大量監視器影像所需的計算成本，以現今的硬體設備還是難以達成

<!--more-->

# Abstract

[DynCNN: An Effective Dynamic Architecture on Convolutional Neural Network for Surveillance Videos](https://openreview.net/forum?id=HyVxPsC9tm) 此篇論文主要研究內容希望利用連續影像的場景具有高相似度的特性，改善傳統 CNN 的運算架構，並適時裁減神經網路的權重參數以減少冗餘的運算量，從而降低功率並達到加速目的，使得系統能在現今的高端硬體設備上，更有效率地處理多通道的大量監控影像

監控影像系統中，由於監控攝影設備是被固定在天花板或是牆壁上，所拍攝出來的影像都是在同一個場景，因此影像之間的背景具有高相似度的特性，在不考慮其它外在因素下會發現影像場景的變化佔整張影像平均不到 30%，如下圖

<figure align="center">
    <img src="https://i.imgur.com/x6589qS.png">
    <figcaption>智慧監控影像</figcaption>
</figure>

在處理這種高相似性的連續影像通常會將整張影像一起計算，所以此研究針對此特性改善傳統 CNN 架構，改善後的架構只針對有改變的影像做卷積，從而達到減少運算量並降低功率加快速度的目標

# Method

## 動態卷積神經網路模型

作者提出了動態卷積神經網路架構(Dynamic Convolutional Neural Networks, DynCNN)，能夠根據每一層的**內部差異圖**(Inner Difference Map, iDM)重新運算擷取新的特徵，並保留位變化區塊的特徵其流程大致如下：

- 首先會對 Frame<sub>t</sub> 與 Frame<sub>t - 1</sub> 使用**幀差法**(Frame Differencin)得到**輸入差異圖**(Input Difference Map, IDM)
- 利用膨脹運算子(Dilation Operator)推導出第一層的內部差異圖(1<sup>st</sup> iDM<sub>t</sub>)，再藉由內部差異圖上的標記決定該層特徵圖(Feature Map, FM)上那些特徵值需要被重新計算
- 往後的每一層內部差異圖(n<sup>th</sup> iDM<sub>t</sub>)也藉由前一層內部差異圖((n - 1)<sup>th</sup> iDM<sub>t</sub>)做膨脹運算推導出來
- Frame<sub>t</sub> 的每一層特徵圖(n<sup>th</sup> FM<sub>t</sub>)會先透過對應層的內部差異圖(n<sup>th</sup> iDM<sub>t</sub>)上的資訊得知需要被更新的特徵值，此去控制 Frame<sub>t</sub> 的 (n - 1)<sup>th</sup> FM<sub>t</sub> 的哪些區塊需要做卷積運算，並將卷積後的結果代入到上一幀的該層特徵圖 n<sup>th</sup> FM<sub>t-1</sub> 做更新的動作來生成當前幀的該層特徵圖

<figure align="center">
    <img src="https://i.imgur.com/gWzPkHA.png">
    <figcaption>動態卷積神經網路架構</figcaption>
</figure>

### 輸入差異圖

透過幀差法獲得，此方法通常被應用在運動目標檢測和分割，原理為在 image sequence 中的相鄰兩幀採用基於像素的時間差分，對相對應的像素點相減再通過二值化來提取影像中的運動區域，通常差分途可以表示為：

$$ \Delta I(i, j) = I_{curr}(i, j) - I_{prev}(i, j) $$

此式子用來計算相鄰兩幀之間影像強度的差異，由於影像中有許多訊干擾，像是鏡頭斑點雜訊、環境光線變化等等，在亮度變化不大的情況下可以藉由設定**閾值**(Threshold)來過濾這些雜訊的干擾，如果對應像素值得變化於是先設定的閾值，則可以認為以處為背景像素，判斷式如下：

$$ D(i, j)= \begin{cases} 1, & \text{if } \vert \Delta I(i, j) \vert > \Theta_{IDM} \\\ 0, & \text{otherwise} \end{cases} $$

其中 $  \Theta_{IDM} $ 代表閾值，$ D(i, j) $ 則表示影像中發生變化的區域，該公式預設的閾值大小很重要，因為會影響準確度和 inference 速度，當閾值過大時有可能會忽略掉許多真正的**變化點**(Change Point)，反之則會造成過多不必要的計算

### 內部差異圖

在卷積過程中，以大小為 3 x 3 的卷積核為例，每個特徵圖上的一個點都是由上一層所對應的九個點來決定，當上一層的某個點改變時隨即牽連到下一層的九個點，如下圖所示，這個過程稱之為**擴散效應**，由上一段所提到的輸入差異圖可以得知那些像素點有改變，代表下一層特徵圖對應的特徵點本身及周圍都會被影響，因此此架構使用內部差異圖來記錄這些被影響的特徵點為 **Impacted Points** 來表示需要被更新的點

<figure align="center">
    <img src="https://i.imgur.com/UiiYLwC.png">
    <figcaption>3 x 3 卷積核進行卷積過程</figcaption>
</figure>

基於擴散效應，每層內部差異圖可以透過對上一層的差異圖(內部差異圖或輸入差異圖，簡稱為差異圖)做膨脹運算來推算被影響的像素點位置

$$ A \oplus B = {x \vert B_x \cap A \neq 0} $$

在此研究中輸入集合 $ A $ 代表差異圖，結構元素 $ B $ 則代表卷積核的形狀，以一個兩層的 CNN 為例，第一層內部差異圖會根據輸入差異圖上的標記合當前卷積核的擴散效應去決定需要被更新的位置，之後每一層內部差異圖也都會根據前一層的內部差異圖來決定需被更新的位置

<figure align="center">
    <img src="https://i.imgur.com/ecrzJ10.png">
    <figcaption>經膨脹運算產生的內部差異圖</figcaption>
</figure>

### 動態卷積

內部差異圖記錄哪些需要被更新的特徵點，這些需要被更新的特徵點，其對應前一層所需的像素點(Needed Pixels)可以透過該層卷積核大小資訊，並結合膨脹運算子推出來，如下圖所示

<figure align="center">
    <img src="https://i.imgur.com/bSDEvoX.png">
    <figcaption>對應所需的像素點</figcaption>
</figure>

由上圖可以看出動態卷積行為是特定像素運算而非連續區域的卷積運算，這對於一般所使用的神經網路加速函式庫的方式是不同的，例如 cudnn 其所提供的卷積無法指定特定的像素點做計算，為了減少不必要的計算作者將 Impacted Pixels 與對應的 Needed Pixels 直接進行資料搬移至另一個連續的記憶體空間，這種將原本特徵圖映射至另一段記憶體空間進而做卷積運算的方式即為動態卷積

<figure align="center">
    <img src="https://i.imgur.com/Bb4f7JX.png">
    <figcaption>特徵點資料搬移示意圖</figcaption>
</figure>

這邊以數字來量化研究成果，若以一張 15 x 15 的監控影像來做為輸入，使用一層的卷積網路架構且與 4 個 3 x 3 的卷積核做運算，總運算量高達 8.1k 的每秒浮點數運算次數(Floating-point operations per second, FLPOS)，但如果場景中變化的只有 30 個像素點，經由提出架構可以讓運算量降至 1.08k FLOPS，與一般卷積的運算量相差約 8.43 倍
