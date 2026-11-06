# Turns standalone HTML pages uploaded through GitHub into site entries.
#
#   research/<name>.html  or  research/<name>/index.html  -> Research page list
#                                                           (site.data["research_pages"])
#   posts/<name>.html     or  posts/<name>/index.html     -> blog post list
#                                                           (a post that links to the page)
#
# Bilingual uploads: a folder may also hold ko.html (and/or en.html) next to
# index.html, e.g. posts/my-study/index.html (English) + posts/my-study/ko.html
# (Korean). Each file becomes its own entry; its language is read from
# <html lang="..."> (default: the site language). Entries from the same folder
# share the ref <name>, so _plugins/i18n.rb links them as translations.
#
# Title comes from <title>, summary from <meta name="description">, date from
# <meta name="date" content="YYYY-MM-DD"> (newest first).
require "cgi"
require "time"

module UploadedHtml
  class Generator < Jekyll::Generator
    safe true
    # Must run before pagination/archives read site.posts (same as al_ext_posts).
    priority :high

    def generate(site)
      site.data["research_pages"] = collect(site, "research").sort_by { |i| [i["date"], i["title"]] }.reverse

      posts = site.collections["posts"]
      collect(site, "posts").each do |item|
        date = parse_date(item["date"]) || site.time
        slug = Jekyll::Utils.slugify(item["name"])
        slug = "html-post" if slug.empty?
        slug = "#{slug}-#{item['lang']}" if item["variant"]
        doc = Jekyll::Document.new(
          site.in_source_dir("_posts/#{date.strftime('%Y-%m-%d')}-#{slug}.md"),
          { :site => site, :collection => posts }
        )
        doc.data["layout"] = "post"
        doc.data["title"] = item["title"]
        doc.data["description"] = item["description"]
        doc.data["date"] = date
        doc.data["redirect"] = item["url"]
        doc.data["lang"] = item["lang"]
        doc.data["ref"] = "upload-#{item['name']}"
        doc.data["tags"] = []
        doc.data["categories"] = []
        doc.content = ""
        posts.docs << doc
      end
      posts.docs.sort!
    end

    private

    def collect(site, folder)
      pattern = %r{\A/?#{folder}/([^/]+?)(?:/(index|ko|en))?\.html\z}
      default_lang = site.config["lang"] || "en"
      site.static_files.filter_map do |file|
        match = file.relative_path.match(pattern)
        next unless match
        next if match[1] == "index" && match[2].nil?

        html = File.read(file.path, encoding: "UTF-8")
        lang = html[/<html[^>]*\slang=["']([a-zA-Z-]+)["']/i, 1].to_s.downcase[0, 2]
        lang = (match[2] == "ko" || match[2] == "en") ? match[2] : default_lang if lang.empty?
        {
          "name" => match[1],
          "variant" => %w[ko en].include?(match[2]),
          "lang" => lang,
          "ref" => "upload-#{match[1]}",
          "title" => clean(html[%r{<title[^>]*>(.*?)</title>}im, 1]) || match[1].tr("-_", "  "),
          "description" => clean(meta(html, "description")),
          "date" => meta(html, "date").to_s.strip,
          "url" => file.url.sub(%r{index\.html\z}, ""),
        }
      end
    end

    def meta(html, name)
      tag = html[/<meta[^>]+name=["']#{name}["'][^>]*>/i]
      tag && tag[/content=["']([^"']*)["']/i, 1]
    end

    def clean(text)
      return nil if text.nil?
      cleaned = CGI.unescapeHTML(text.gsub(/\s+/, " ").strip)
      cleaned.empty? ? nil : cleaned
    end

    def parse_date(text)
      return nil if text.nil? || text.empty?
      Time.parse(text)
    rescue ArgumentError
      nil
    end
  end
end
