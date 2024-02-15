---
title: "Diffusion Model 學習筆記 - DDPM"
slug: ddpm
date: 2024-02-15T11:30:32+08:00
categories:
- deep learning
- diffusion model
tags:
- ddpm
thumbnailImage: https://i.imgur.com/iMe5Dqm.png
---

玩了 Stable Diffusion 一陣也該是時後來了解一下其生成圖片的原理了，跟過去最火的 GAN 比起來真的複雜很多，但不得不說生出來的圖片效果真的非常好，Open AI 的 DALL·E 跟 Google Imagen 背後也都是使用 Diffusion Model，就讓我們來一探究竟吧

<!--more-->

{{< alert info >}}
本文為李宏毅老師 [【生成式AI】Diffusion Model 原理剖析](https://youtu.be/ifCDXFdeaaM?si=K3iT_cVIaIwKIs7x) 的上課筆記，會著重在 Diffusion Model 中反向過程的原理推導，這邊假設讀者已經知道前向過程 (Diffusion Process)，並且知道 $ p $、$ q $ 分布的涵義為何
{{< /alert >}}

生成模型的本質就是從 Latent space 中取樣出一組雜訊，將雜訊輸入到生成模型中會輸出一組資料分布，希望可以近似真實資料的分布，而這時候通常會使用 Maximum Likelihood Estimation (MLE)。

<figure align="center">
    <img src="https://i.imgur.com/1IMYurm.png">
</figure>

其中 $ P_{data} $ 代表真實資料的機率分布，$ \theta^* $ 表示優化後的模型使得生成 $ x^1, ..., x^m $ 的機率連乘為最大。

<figure align="center">
    <img src="https://i.imgur.com/7gsY7N1.png">
</figure>

* 第一行加入 $ \log $ 後可以將連乘改為總和，因為不會影響 $ \mathop{\arg\max}\limits_{\theta} $ 的結果所以可以畫上等號

* 第二行 summation 的部分可以近似於取 $ x $ 從 $ P_{data} $ 中 sample 出來的期望值

* 第三行減掉 $ \int\limits_{x} P_{data}(x) \log P_{data}(x) dx $ 方便讓我們把式子轉為 KL Divergence 的形式

因此我們可知 maximum likelihood 等價於 minimize KL Divergence，也就是說我們只要想辦法讓兩個分布的差異越小越好即可。

但是要如何計算 $ P_{\theta}(x) $ 呢，根據 VAE 的經驗我們沒辦法去直接計算，但是我們可以去計算 lower bound of $ \log P_{\theta}(x) $，我們只要去 maximize 這個 lower bound 就好，推導過程如下:

<figure align="center">
    <img src="https://i.imgur.com/X9L6ylt.png">
</figure>

* 其中 $ q(z|x) $ 可以對應至 Diffusion Model 中前向過程的 $ q(x_{1:T}|x_0) $

* 至於為甚麼可以加入 $ q(z|x) $，這是一個令人匪夷所思的操作，它可以是任意一種 distribution，都可以滿足這個等式

* 根據貝氏定理可以把 $ P(x) $ 展開

* 橘色大於等於 0 的項是因為它是 KL Divergence 所以不會有小於 0 的情況

接著 lower bound of $ \log P_{\theta}(x) $ 經過化簡最終可以寫成三個項:
<figure align="center">
    <img src="https://i.imgur.com/lrjmtkD.png">
    <img src="https://i.imgur.com/EzamlGT.png">
</figure>

<figure align="center">
    <img src="https://i.imgur.com/du6qNgm.png">
</figure>

前兩項因為跟模型要學習的參數 $ \theta $ 無關顧可以當作已知，所以我們要計算的只有紅色中的式子，而這項恰好又是一個 KL Divergence，我們得去計算在 given $ x_{t} $ 的情況下 denoise 為 $ x_{t-1} $ 的機率分布為何，$ P_{\theta} $ 分布是模型要去學習的。雖然我們不知道 $ q(x_{t-1} | x_t) $ 怎麼計算，但如果有多給定 $ x_0 $，我們一樣可以使用貝氏定理將其展開:

<figure align="center">
    <img src="https://i.imgur.com/R3PRx8R.png">
</figure>

如圖，由於三項都是已知，便可以把 Gaussian function 寫出來進行化簡，這邊須注意的是過程需用到 Gaussian distribution 的性質才可以，如果只是任意 distribution 是無法搞定它的，化簡過程如下:

<figure align="center">
    <img src="https://i.imgur.com/M5CSEwr.png">
</figure>

化簡之後發現結果也符合 Gaussian distribution，左邊那項是該分布的 mean 而右邊是 variance，這邊我覺得李宏毅老師講得非常好，我們可以去觀察它的 mean，對 $ x_0 $ 與 $ x_t $ 各乘上一個常數(看作是權重)，相加後再除以一個常數，那 $ x_{t-1} $ 不就可以看做是 $ x_0 $ 與 $ x_t $ 做某種 interpolation 後的產物嗎，這麼一想就非常直觀了。

<figure align="center">
    <img src="https://i.imgur.com/Plpbpdo.png">
</figure>

到這邊應該眼睛都看瞎了，只剩最後一步，要怎麼算這兩個分布的 KL Divergence 呢，其實它是有公式解的，但在 DDPM 中使用更簡單的方法，也就是固定它們的 variance，這樣一來只要去比較他們的 mean 就可以了，作者有提到他們也有嘗試去讓模型學習 variance 但效果沒有比較好所以乾脆就讓它固定。

<figure align="center">
    <img src="https://i.imgur.com/JN92CU5.png">
</figure>

也就是說在做 sampling 時，當我們給定 $ x_t $ 時模型真正要去預測的其實是一個 Gaussian distribution 的 mean，$ x_t $ 減掉符合這個 distribution 的雜訊後就可以得到 $ x_{t-1} $，使用迴圈一步一步推回至 $ x_0 $，大功告成！

<figure align="center">
    <img src="https://i.imgur.com/qUbXsWA.png">
</figure>

至於為甚麼紅框中可以寫成全部都是 $ x_t $ 的形式，不是還有 $ x_0 $ 嗎，這是因為我們知道可以將前向過程寫成一個 closed-form，移項移一移就可以了，如下:

<figure align="center">
    <img src="https://i.imgur.com/v2IfJS2.png">
</figure>