---
title: "Winograd algorithm"
slug: winograd-algorithm
date: 2022-04-12T16:53:08+08:00
categories:
- deep learning
- cnn
tags:
- winograd algorithm
- acceleration
#thumbnailImage: //example.com/image.jpg
---

Winograd 演算法可以減少矩陣乘法中的乘法運算，近年來有許多相關研究將其應用於加速 convolutional operation

<!--more-->

## 1D Winograd

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

如果是一般的矩陣乘法則有 6 次的乘法計算及 4 次加法，但輸入資訊轉換成矩陣並不是任意矩陣，而是有規律地分布著大量重複 element 的矩陣，那麼就可以透過 Winograd 演算法寫成以下形式：

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

## 矩陣化

接下來我們將一維卷積公式推廣成矩陣的形式：

$$ Y = A ^ T[(Gg) \bigodot (B ^ T d)] $$