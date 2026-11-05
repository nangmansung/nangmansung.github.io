// Floating "+" button (bottom right) with: new post, upload HTML, download template.
// Shown only in browsers that opened /write/ once (sets localStorage "nm-owner"),
// so ordinary visitors never see it.
(function () {
  var REPO = "https://github.com/nangmansung/nangmansung.github.io";

  function pad(n) {
    return String(n).padStart(2, "0");
  }

  function newPostUrl() {
    var d = new Date();
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
      "",
    ].join("\n");
    return (
      REPO + "/new/main/_posts?filename=" + encodeURIComponent(today + "-new-post.md") + "&value=" + encodeURIComponent(template)
    );
  }

  window.nmNewPostUrl = newPostUrl;
  window.nmUploadUrl = REPO + "/upload/main/research";
  window.nmTemplateUrl = "/assets/templates/blog-template.html";

  function isOwner() {
    try {
      return localStorage.getItem("nm-owner") === "1";
    } catch (e) {
      return false;
    }
  }

  function build() {
    if (!isOwner() || document.querySelector(".nm-fab")) return;

    var wrap = document.createElement("div");
    wrap.className = "nm-fab";
    wrap.innerHTML =
      '<div class="nm-fab-menu" role="menu">' +
      '<a role="menuitem" target="_blank" rel="noopener" class="nm-fab-item" data-act="post">✏️ 새 글 쓰기</a>' +
      '<a role="menuitem" target="_blank" rel="noopener" class="nm-fab-item" data-act="upload">⬆️ HTML 업로드</a>' +
      '<a role="menuitem" class="nm-fab-item" data-act="template" download="blog-template.html">⬇️ 기본 HTML 다운로드</a>' +
      "</div>" +
      '<button type="button" class="nm-fab-btn" aria-label="글쓰기 메뉴" aria-expanded="false">+</button>';
    document.body.appendChild(wrap);

    wrap.querySelector('[data-act="post"]').href = newPostUrl();
    wrap.querySelector('[data-act="upload"]').href = window.nmUploadUrl;
    wrap.querySelector('[data-act="template"]').href = window.nmTemplateUrl;

    var btn = wrap.querySelector(".nm-fab-btn");
    function setOpen(open) {
      wrap.classList.toggle("nm-fab-open", open);
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    }
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      setOpen(!wrap.classList.contains("nm-fab-open"));
    });
    document.addEventListener("click", function (e) {
      if (!wrap.contains(e.target)) setOpen(false);
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") setOpen(false);
    });
    wrap.querySelectorAll(".nm-fab-item").forEach(function (a) {
      a.addEventListener("click", function () {
        setOpen(false);
      });
    });
  }

  window.nmSetOwner = function (on) {
    try {
      if (on) localStorage.setItem("nm-owner", "1");
      else localStorage.removeItem("nm-owner");
    } catch (e) {}
    var old = document.querySelector(".nm-fab");
    if (old) old.remove();
    build();
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
  else build();
})();
