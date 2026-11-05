---
layout: page
title: 글쓰기
permalink: /write/
description: 블로그 글을 쓰거나 Research 페이지를 올리는 곳 (메뉴에는 보이지 않아요)
nav: false
sitemap: false
---

<div class="write-actions">
  <a id="new-post" class="write-btn" href="https://github.com/nangmansung/nangmansung.github.io/new/main/_posts" target="_blank" rel="noopener">✏️ 블로그 글 쓰기</a>
  <a class="write-btn secondary" href="https://github.com/nangmansung/nangmansung.github.io/upload/main/research" target="_blank" rel="noopener">⬆️ Research에 HTML 올리기</a>
</div>

### 블로그 글 쓰기

1. 버튼을 누르면 GitHub 편집기가 글 양식이 채워진 상태로 열려요.
2. 파일 이름의 `new-post` 부분을 영어 제목으로 바꾸세요. 예: `2026-11-05-ring-attractor.md`
3. `title`, `description`과 본문을 쓰고, 오른쪽 위 **Commit changes**를 누르세요.
4. 1~2분 뒤 블로그에 올라가요.

### Research에 HTML 올리기

1. 버튼을 누르면 GitHub 업로드 화면이 열려요.
2. HTML 파일을 끌어다 놓고 **Commit changes**를 누르세요.
   - 파일 하나면 `내-연구.html`처럼 올리면 돼요.
   - 그림이나 CSS 파일이 같이 있으면 **폴더째로** 끌어다 놓으세요. 폴더 안의 첫 화면 파일 이름은 `index.html`이어야 해요.
3. 1~2분 뒤 Research 페이지 목록에 자동으로 나타나요.

목록에 보이는 제목은 HTML의 `<title>`에서, 설명은 `<meta name="description" content="...">`에서 가져와요. `<meta name="date" content="2026-11-05">`를 넣으면 최신순으로 정렬돼요.

<script>
  (function () {
    var d = new Date();
    var pad = function (n) { return String(n).padStart(2, "0"); };
    var today = d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate());
    var template = [
      "---",
      "layout: distill",
      "title: 제목을 쓰세요",
      "description: 한 줄 요약을 쓰세요",
      "tags: blog",
      "date: " + today,
      "",
      "authors:",
      "  - name: nangmansung",
      '    url: "https://github.com/nangmansung"',
      "---",
      "",
      '<div class="tldr" markdown="1">',
      "**TL;DR** 핵심 내용을 한두 문장으로 쓰세요.",
      "</div>",
      "",
      "## 첫 번째 소제목",
      "",
      "여기에 본문을 쓰세요. 수식은 $E = mc^2$ 처럼 쓸 수 있어요.",
      ""
    ].join("\n");
    var url = "https://github.com/nangmansung/nangmansung.github.io/new/main/_posts" +
      "?filename=" + encodeURIComponent(today + "-new-post.md") +
      "&value=" + encodeURIComponent(template);
    document.getElementById("new-post").href = url;
  })();
</script>
