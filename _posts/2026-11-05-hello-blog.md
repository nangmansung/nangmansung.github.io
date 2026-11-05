---
title: 블로그를 시작합니다
date: 2026-11-05 18:00:00 +0900
categories: [Blog, 공지]
tags: [시작, 사용법]
math: true
---

첫 글입니다. 앞으로 연구하고 공부한 내용을 여기에 기록하려고 해요.

## 글 쓰는 법 (메모)

`_posts` 폴더에 `YYYY-MM-DD-제목.md` 이름으로 파일을 만들면 글이 됩니다.

### 수식

문장 안에 수식 넣기: $E = mc^2$

따로 한 줄에 수식 넣기:

$$
\tau \frac{dV}{dt} = -(V - V_{rest}) + R I(t)
$$

### 코드

```python
import numpy as np
x = np.linspace(0, 2 * np.pi, 100)
print(np.sin(x).max())
```

### 강조 상자

> 이렇게 팁 상자를 만들 수 있어요.
{: .prompt-tip }

> 주의할 내용은 이렇게 써요.
{: .prompt-warning }
