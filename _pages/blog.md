---
layout: profile
title: "Blog"
permalink: /blog/
author_profile: false
---

<section class="blog-index" aria-labelledby="blog-index-title">
  <header class="news-index-header">
    <p class="publication-eyebrow">NOTES · ESSAYS · UPDATES</p>
    <h1 id="blog-index-title">Blog</h1>
    <p class="news-index-introduction">Research notes, technical essays, and longer project updates.</p>
  </header>

  <div class="blog-list">
    {% for post in site.data.blogs %}
      <article class="blog-card">
        <time datetime="{{ post.date }}">{{ post.date }}</time>
        <div>
          <h2>{% if post.url %}<a href="{{ post.url }}">{{ post.title }}</a>{% else %}{{ post.title }}{% endif %}</h2>
          {% if post.summary %}<p>{{ post.summary }}</p>{% endif %}
          {% if post.tags %}<div class="work-tags" aria-label="Blog topics">{% for tag in post.tags %}<span>{{ tag }}</span>{% endfor %}</div>{% endif %}
        </div>
      </article>
    {% endfor %}
  </div>
</section>
