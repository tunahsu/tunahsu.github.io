---
title: "Winograd algorithm 卷積神經網路中的加速卷積算法"
slug: winograd-algorithm
date: 2022-04-12T16:53:08+08:00
categories:
- deep learning
- cnn
tags:
- winograd algorithm
- acceleration
thumbnailImage: https://s2.ax1x.com/2019/05/22/VpBFc6.png
---

作為首篇學習筆記，來記錄一下最近閱讀學長論文時文中的 Winograd 演算法，該方法可以減少矩陣乘法中的乘法運算，近年來有許多相關研究將其應用於加速 convolutional operation

<!--more-->

在查了許多有關 Winograd 演算法的介紹後，發現大多數的文章都只有列出公式跟矩陣，並詳細說明其作用以及從何而來，所以這篇筆記將會參考網路上的文章，用簡單的例子來說明其原理，再針對一般化公式中的各個矩陣做解釋

# 1D Winograd

以一維的卷積為例，輸入資訊為 $ d = \begin{bmatrix} d_0 & d_1 & d_2 & d_3 \end{bmatrix} ^ T $，卷積核為 $ g = \begin{bmatrix} g_0 & g_1 & g_2 \end{bmatrix} ^ T $ ，那麼 $ F(2, 3) $ 的卷積可以寫成以下形式：

$$
    F(2, 3) = 
    \begin{bmatrix}
        d_0 & d_1 & d_2 \\\
        d_1 & d_2 & d_3
    \end{bmatrix}
    \begin{bmatrix}
        g_0 \\\ g_1 \\\ g_2
    \end{bmatrix}
    = 
    \begin{bmatrix}
        d_0 g_0 + d_1 g_1 + d_2 g_2 \\\
        d_1 g_0 + d_2 g_1 + d_3 g_2
    \end{bmatrix}
    =
    \begin{bmatrix}
        r_0 \\\ r_1
    \end{bmatrix}
$$

如果是一般的矩陣乘法則有 6 次的乘法計算及 4 次加法，但輸入資訊轉換成矩陣並不是任意矩陣，而是有規律地分布著大量重複元素的矩陣，那麼就可以透過 Winograd 演算法寫成以下形式：

$$
    F(2, 3) = 
    \begin{bmatrix}
        d_0 & d_1 & d_2 \\\
        d_1 & d_2 & d_3
    \end{bmatrix}
    \begin{bmatrix}
        g_0 \\\ g_1 \\\ g_2
    \end{bmatrix}
    = 
    \begin{bmatrix}
        m_1 + m_2 + m_3 \\\
        m_2 - m_3 - m_4
    \end{bmatrix}
$$

其中，

$$ m_1 = (d_0 - d_2) g_0 $$
$$ m_2 = (d_1 + d_2) \frac{g_0 + g_1 + g_2}{2} $$
$$ m_3 = (d_2 - d_1) \frac{g_0 - g_1 + g_2}{2} $$
$$ m_4 = (d_1 - d_3) g_2 $$

由於在卷積運算中，卷積核中的元素為固定的，所以有關卷積核的運算只需一次，可以被忽略，故乘法的次數可降為 4

# 矩陣化

接下來我們將一維卷積公式推廣成矩陣的形式：

$$ Y = A ^ T[(Gg) \odot (B ^ T d)] $$

其中 $ \odot $ 表示 Hadamard product 矩陣中對應位置的元素相乘，以下將會將此公式拆解成三個部分，首先我們對各個符號做解釋：

- $ A ^ T $ 為輸出變換矩陣，大小為 m x (m + r - 1)
- $ G $ 為卷積核變換矩陣，大小為 (m + r - 1) x r
- $ B ^ T $ 為輸入變換矩陣，大小為 (m + r - 1) x (m + r - 1)
- $ d $ 為輸入資訊，大小為 (m + r - 1) x 1
- $ g $ 卷積核，大小為 r x 1

這邊方便推導我們假設 $ m = 2, r = 3 $ ，為了讓輸入矩陣與卷積核矩陣能夠做內積，必須先乘上變換矩陣讓它們維度一致，而變換矩陣中的元素可由上述 $ F(2, 3) $ 的例子中各項係數得知

$$
    Gg = 
    \begin{bmatrix}
        1 & 0 & 0 \\\
        \frac{1}{2} & \frac{1}{2} & \frac{1}{2} \\\
        \frac{1}{2} & \frac{-1}{2} & \frac{1}{2} \\\
        0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
        g_0 \\\ g_1 \\\ g_2
    \end{bmatrix}
    = 
    \begin{bmatrix}
        g_0 \\\
        \frac{g_0 + g_1 + g_2}{2} \\\
        \frac{g_0 - g_1 + g_2}{2} \\\
        g_2
    \end{bmatrix}
$$

$$
    B ^ T d = 
    \begin{bmatrix}
        1 & 0 & -1 & 0 \\\
        0 & 1 & 1 & 0 \\\
        0 & -1 & 1 & 0 \\\
        0 & 1 & 0 & -1
    \end{bmatrix}
    \begin{bmatrix}
        d_0 \\\ d_1 \\\ d_2 \\\ d_3
    \end{bmatrix}
    = 
    \begin{bmatrix}
        d_0 - d_2 \\\
        d_1 + d_2 \\\
        d_2 - d_1 \\\
        d_1 - d_3
    \end{bmatrix}
$$

將以上兩項做內積我們就可以得到一向量為 $ \begin{bmatrix} m_1 & m_2 & m_3 & m_4 \end{bmatrix} ^ T $ ，最後在將其左乘 $ A ^ T $ 可得

$$
    Y = A ^ T
    \begin{bmatrix}
        m_1 \\\
        m_2 \\\
        m_3 \\\
        m_4
    \end{bmatrix}
    =
    \begin{bmatrix}
        1 & 1 & 1 & 0 \\\
        0 & 1 & -1 & -1 \\\
    \end{bmatrix}
    \begin{bmatrix}
        m_1 \\\
        m_2 \\\
        m_3 \\\
        m_4
    \end{bmatrix}
    =
    \begin{bmatrix}
        m_1 + m_2 + m_3 \\\
        m_2 - m_3 - m_4
    \end{bmatrix}
$$

從矩陣關係以及其大小我們可以看出原本需要 m x r 次的乘法可以降為 m + r - 1 次，在兩者都很大的情況下可以得到很大的效能提升，至於一維以上的情況可以參考[详解卷积中的Winograd加速算法](https://zhuanlan.zhihu.com/p/260109670)