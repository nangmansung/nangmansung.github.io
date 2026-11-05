# Lists the standalone HTML pages uploaded under research/ so _pages/research.md
# can show them. Supported shapes:
#   research/<name>.html
#   research/<name>/index.html   (a folder, e.g. with its own css/ and images)
# Title comes from <title>, summary from <meta name="description">, and an
# optional <meta name="date" content="YYYY-MM-DD"> sets the order (newest first).
require "cgi"

module ResearchPages
  class Generator < Jekyll::Generator
    safe true
    priority :low

    PATTERN = %r{\A/?research/([^/]+?)(/index)?\.html\z}

    def generate(site)
      items = site.static_files.filter_map do |file|
        match = file.relative_path.match(PATTERN)
        next unless match
        next if match[1] == "index" && match[2].nil?

        html = File.read(file.path, encoding: "UTF-8")
        title = meta_text(html[%r{<title[^>]*>(.*?)</title>}im, 1]) || match[1].tr("-_", "  ")
        url = file.url.sub(%r{index\.html\z}, "")
        {
          "title" => title,
          "description" => meta_text(meta(html, "description")),
          "date" => meta(html, "date").to_s.strip,
          "url" => url,
        }
      end
      site.data["research_pages"] = items.sort_by { |i| [i["date"], i["title"]] }.reverse
    end

    private

    def meta(html, name)
      tag = html[/<meta[^>]+name=["']#{name}["'][^>]*>/i]
      tag && tag[/content=["']([^"']*)["']/i, 1]
    end

    def meta_text(text)
      return nil if text.nil?
      cleaned = CGI.unescapeHTML(text.gsub(/\s+/, " ").strip)
      cleaned.empty? ? nil : cleaned
    end
  end
end
