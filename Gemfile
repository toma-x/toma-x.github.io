source "https://rubygems.org"

# Jekyll version (used latest stable version)
gem "jekyll", "~> 4.3.2"

# Plugins mentioned in _config.yml
gem "jekyll-paginate", "~> 1.1"

# Other common Jekyll plugins
gem "jekyll-feed", "~> 0.17.0"
gem "jekyll-seo-tag", "~> 2.8.0"

# Windows and JRuby does not include zoneinfo files, so bundle the tzinfo-data gem
# and associated library.
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", "~> 2.0"
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]

# Lock `http_parser.rb` gem to `v0.6.0` on JRuby builds since newer versions of the gem
# do not have a Java counterpart.
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]

# For live-reloading during development
group :development do
  gem "webrick", "~> 1.8"
end 