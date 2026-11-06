# Two-language site (English default, Korean under /ko/).
#
# Every page and post has a language: front matter `lang: en|ko`
# (default: site `lang`). Pages/posts that are translations of each other
# share the same `ref` (default for posts: their slug; uploaded HTML: the
# folder name). This plugin sets, on every page, post and Research item:
#
#   translations : { "ko" => "/ko/blog/", ... }  other-language versions
#   list_langs   : languages whose lists should show this entry - its own
#                  language, plus any language it has no translation in
#                  (shown there with a "KOR only" / "ENG only" badge)
#
# Lists filter with:  where_exp: "p", "p.list_langs contains page.lang"
module SiteI18n
  LANGS = %w[en ko].freeze

  class Generator < Jekyll::Generator
    safe true
    priority :normal   # after UploadedHtml (:high) has added the uploaded posts

    def generate(site)
      default = site.config["lang"] || "en"
      entries = site.pages.select { |p| p.data["ref"] } + site.posts.docs
      entries.each do |e|
        e.data["lang"] = (e.data["lang"] || default).to_s
        e.data["ref"] ||= e.data["slug"] || File.basename(e.path.to_s, ".*")
      end
      annotate(entries.map { |e| [e.data, e.url] })

      research = site.data["research_pages"] || []
      research.each { |i| i["lang"] ||= default }
      annotate(research.map { |i| [i, i["url"]] })
    end

    private

    # items: [[data_hash, url], ...]; data_hash has "lang" and "ref"
    def annotate(items)
      by_ref = Hash.new { |h, k| h[k] = {} }
      items.each { |d, url| by_ref[d["ref"]][d["lang"]] ||= url }
      items.each do |d, _url|
        group = by_ref[d["ref"]]
        d["translations"] = group.reject { |l, _| l == d["lang"] }
        d["list_langs"] = [d["lang"]] + (LANGS - group.keys)
      end
    end
  end
end
