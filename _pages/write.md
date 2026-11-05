---
layout: page
title: 글쓰기
permalink: /write/
description: 블로그 글을 쓰거나 HTML 페이지를 올리는 곳 (메뉴에는 보이지 않아요)
nav: false
sitemap: false
---

<div id="write-lock">
  <p>비밀번호를 입력하면 이 브라우저에서 글쓰기 메뉴와 오른쪽 아래 <b>+</b> 버튼이 켜져요.</p>
  <form id="unlock-form" class="write-actions" autocomplete="off">
    <input id="unlock-pw" type="password" placeholder="비밀번호" aria-label="비밀번호"
      style="padding: 9px 12px; border: 1px solid var(--global-divider-color); border-radius: 6px; min-width: 220px;">
    <button type="submit" class="write-btn" style="border: none;">확인</button>
  </form>
  <p id="unlock-msg" style="color: #d93a4b;"></p>
</div>

<div id="write-panel" style="display: none;">
  <div class="write-actions">
    <a id="new-post" class="write-btn" href="#" target="_blank" rel="noopener">✏️ 새 글 쓰기</a>
    <a id="upload-blog" class="write-btn secondary" href="#" target="_blank" rel="noopener">⬆️ blog에 HTML 업로드</a>
    <a id="upload-research" class="write-btn secondary" href="#" target="_blank" rel="noopener">⬆️ Research에 HTML 업로드</a>
    <a class="write-btn secondary" href="/assets/templates/blog-template.html" download="blog-template.html">⬇️ 기본 HTML 다운로드</a>
  </div>

  <p>이 브라우저에서는 모든 페이지 오른쪽 아래에 <b>+</b> 버튼이 보여요. <a href="#" id="owner-off">이 브라우저에서 잠그기</a></p>

  <h3>새 글 쓰기</h3>
  <ol>
    <li>버튼을 누르면 GitHub 편집기가 글 양식이 채워진 상태로 열려요.</li>
    <li>파일 이름의 <code>new-post</code> 부분을 영어 제목으로 바꾸세요. 예: <code>2026-11-05-ring-attractor.md</code></li>
    <li><code>title</code>, <code>description</code>과 본문을 쓰고, 오른쪽 위 <b>Commit changes</b>를 누르세요.</li>
    <li>1~2분 뒤 블로그에 올라가요.</li>
  </ol>

  <h3>HTML 업로드</h3>
  <ol>
    <li><b>기본 HTML 다운로드</b>로 양식을 받아 내용을 바꿔 쓰세요. <code>&lt;title&gt;</code>, 설명, 날짜(<code>&lt;meta name="date"&gt;</code>)도 꼭 바꿔 주세요.</li>
    <li>+ 버튼의 <b>HTML 업로드</b>는 지금 보고 있는 페이지에 따라 올라가는 곳이 달라요.
      <ul>
        <li>blog나 홈에서 누르면 → <b>blog 글 목록</b>에 추가 (<code>posts</code> 폴더)</li>
        <li>Research에서 누르면 → <b>Research 목록</b>에 추가 (<code>research</code> 폴더)</li>
      </ul>
    </li>
    <li>파일을 끌어다 놓고 <b>Commit changes</b>를 누르면 1~2분 뒤 목록에 나타나요. 그림이 있으면 폴더째 올리고, 폴더 안 HTML 이름은 <code>index.html</code>로 해주세요.</li>
  </ol>
</div>

<script>
  document.addEventListener("DOMContentLoaded", function () {
    var HASH = "b6b24d39d00974452464b4a6237e9045bb8ffa066af61ee7932223a2e440c9f7";
    var lock = document.getElementById("write-lock");
    var panel = document.getElementById("write-panel");

    function owner() {
      try { return localStorage.getItem("nm-owner2") === "1"; } catch (e) { return false; }
    }
    function show() {
      lock.style.display = "none";
      panel.style.display = "block";
      if (window.nmNewPostUrl) document.getElementById("new-post").href = window.nmNewPostUrl();
      if (window.nmUploadUrl) {
        document.getElementById("upload-blog").href = window.nmUploadUrl("posts");
        document.getElementById("upload-research").href = window.nmUploadUrl("research");
      }
    }
    async function sha256(text) {
      var buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
      return Array.from(new Uint8Array(buf)).map(function (b) { return b.toString(16).padStart(2, "0"); }).join("");
    }

    if (owner()) show();

    document.getElementById("unlock-form").addEventListener("submit", async function (e) {
      e.preventDefault();
      var pw = document.getElementById("unlock-pw").value.trim();
      if ((await sha256(pw)) === HASH) {
        window.nmSetOwner(true);
        show();
      } else {
        document.getElementById("unlock-msg").textContent = "비밀번호가 달라요.";
      }
    });

    document.getElementById("owner-off").addEventListener("click", function (e) {
      e.preventDefault();
      window.nmSetOwner(false);
      panel.style.display = "none";
      lock.style.display = "block";
      document.getElementById("unlock-msg").textContent = "";
    });
  });
</script>
