---
layout: page
title: 글쓰기
permalink: /write/
description: 블로그 글을 쓰거나 Research 페이지를 올리는 곳 (메뉴에는 보이지 않아요)
nav: false
sitemap: false
---

<div class="write-actions">
  <a id="new-post" class="write-btn" href="https://github.com/nangmansung/nangmansung.github.io/new/main/_posts" target="_blank" rel="noopener">✏️ 새 글 쓰기</a>
  <a class="write-btn secondary" href="https://github.com/nangmansung/nangmansung.github.io/upload/main/research" target="_blank" rel="noopener">⬆️ HTML 업로드</a>
  <a class="write-btn secondary" href="/assets/templates/blog-template.html" download="blog-template.html">⬇️ 기본 HTML 다운로드</a>
</div>

<p id="owner-status"></p>

### 새 글 쓰기

1. 버튼을 누르면 GitHub 편집기가 글 양식이 채워진 상태로 열려요.
2. 파일 이름의 `new-post` 부분을 영어 제목으로 바꾸세요. 예: `2026-11-05-ring-attractor.md`
3. `title`, `description`과 본문을 쓰고, 오른쪽 위 **Commit changes**를 누르세요.
4. 1~2분 뒤 블로그에 올라가요.

### HTML 업로드 (Research)

1. **기본 HTML 다운로드**로 양식 파일을 받아서, 그 안의 안내대로 내용을 바꿔 쓰세요. 블로그 글과 같은 모양으로 나와요.
2. **HTML 업로드**를 누르고 파일을 끌어다 놓은 뒤 **Commit changes**를 누르세요.
   - 파일 하나면 `ring-attractor.html`처럼 올리면 돼요. 영어 파일 이름을 추천해요.
   - 그림 파일이 같이 있으면 **폴더째로** 끌어다 놓으세요. 폴더 안의 HTML 파일 이름은 `index.html`이어야 해요.
3. 1~2분 뒤 Research 페이지 목록에 자동으로 나타나요.

<script>
  document.addEventListener("DOMContentLoaded", function () {
    if (window.nmSetOwner) window.nmSetOwner(true);
    if (window.nmNewPostUrl) document.getElementById("new-post").href = window.nmNewPostUrl();
    var status = document.getElementById("owner-status");
    status.innerHTML =
      '이 브라우저에서는 모든 페이지 오른쪽 아래에 <b>+</b> 버튼이 보여요. ' +
      '<a href="#" id="owner-off">이 브라우저에서 + 버튼 숨기기</a>';
    document.getElementById("owner-off").addEventListener("click", function (e) {
      e.preventDefault();
      window.nmSetOwner(false);
      status.textContent = "+ 버튼을 숨겼어요. 이 페이지에 다시 들어오면 다시 보여요.";
    });
  });
</script>
