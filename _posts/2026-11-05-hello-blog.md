---
layout: distill
title: 블로그를 시작합니다
description: 앞으로 연구하고 공부한 내용을 이곳에 기록합니다. 이 글은 글쓰기 형식 예시도 겸해요.
tags: blog
date: 2026-11-05
featured: true
lang: ko

authors:
  - name: nangmansung
    url: "https://github.com/nangmansung"
    affiliations:
      name: SKKU

bibliography: 2026-11-05-hello-blog.bib
---

<div class="tldr" markdown="1">
**TL;DR** 블로그를 열었습니다. 아래는 수식, 각주, 인용, 코드, 그림 설명을 쓰는 방법 예시예요.
</div>

## 수식

문장 안의 수식은 $E = mc^2$ 처럼 쓰고, 따로 한 줄에 쓰는 수식은 이렇게 써요.

$$
\tau \frac{dV}{dt} = -(V - V_{\text{rest}}) + R\, I(t)
$$

## 각주와 인용

문장 끝에 각주를 달 수 있어요.<d-footnote>각주 내용은 글 맨 아래에 모여서 보여요.</d-footnote>
논문 인용도 할 수 있어요 <d-cite key="hopfield1982"></d-cite>. 참고문헌 목록은 글 맨 아래에 자동으로 만들어져요.

## 코드

```python
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
print(np.sin(x).max())
```

## 그림

그림은 `assets/img/` 폴더에 넣고 이렇게 불러와요.

```html
<figure>
  <img src="/assets/img/my-figure.png" alt="설명">
  <figcaption><b>Figure 1:</b> 그림 설명</figcaption>
</figure>
```
