---
layout: profile
title: "News"
permalink: /news/
author_profile: false
---

<section class="news-index" aria-labelledby="news-index-title">
  <header class="news-index-header">
    <p class="publication-eyebrow">CAREER · RELEASES · PUBLICATIONS · OPEN SOURCE</p>
    <h1 id="news-index-title">All News</h1>
    <p class="news-index-introduction">
      A complete timeline of public research progress, from career transitions and model releases to conference milestones and open-source work.
    </p>
    <div class="news-index-summary" aria-label="News timeline summary">
      <span><strong>{{ site.data.news | size }}</strong> recorded updates</span>
      <span><strong>2019–2026</strong> timeline</span>
      <span><strong>{{ site.data.profile.profile.publication_count }}</strong> Scholar publications</span>
    </div>
  </header>

  {% assign news_years = site.data.news | group_by: "year" %}
  <nav class="news-year-nav" aria-label="Jump to news by year">
    {% for year in news_years %}<a href="#news-{{ year.name }}">{{ year.name }}</a>{% endfor %}
  </nav>

  <div class="news-timeline">
    {% for year in news_years %}
      <section class="news-year-group" id="news-{{ year.name }}" aria-labelledby="news-year-{{ year.name }}">
        <h2 id="news-year-{{ year.name }}">{{ year.name }}</h2>
        <ul class="news-list-v2 news-list-full">
          {% for item in year.items %}
            <li>
              <time datetime="{{ item.date }}">{{ item.label }}</time>
              <div class="news-item-copy">
                <span class="news-category">{{ item.category }}</span>
                <p>{{ item.content }}</p>
              </div>
            </li>
          {% endfor %}
        </ul>
      </section>
    {% endfor %}
  </div>
</section>
