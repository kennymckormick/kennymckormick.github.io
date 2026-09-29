---
permalink: /
title: "Haodong Duan (段浩东)"
excerpt: "Researcher working on multimodal learning and LLM/LMM evaluation"
layout: profile
author_profile: false
redirect_from:
  - /about/
  - /about.html
---

<section class="profile-intro" aria-labelledby="profile-name">
  <h1 id="profile-name">Haodong Duan</h1>
  <p class="profile-role">Researcher · Multimodal Learning &amp; Evaluation</p>
  <p class="profile-affiliation"><strong>ByteDance Seed</strong><span aria-hidden="true">·</span> Singapore</p>
  <p class="profile-email"><i class="fas fa-envelope" aria-hidden="true"></i><a href="mailto:dhd.efz@gmail.com">dhd.efz@gmail.com</a></p>

  <div class="collaboration-panel" aria-label="Collaboration interests">
    <p class="collaboration-note">欢迎围绕多模态学习、大模型评测与视频理解开展学术合作，也欢迎对开源评测工具和基准感兴趣的研究者与工程师联系我。</p>
    <p class="collaboration-note">I am open to academic collaborations on multimodal learning, LLM/LMM evaluation, and video understanding. Please feel free to reach out by email.</p>
  </div>

  <h2 class="section-heading">Research Interests</h2>
  <p class="research-copy">
    My research focuses on <strong>multimodal learning</strong>, <strong>LLM/LMM evaluation</strong>, and <strong>video understanding</strong>.
    I build open-source evaluation infrastructure and benchmarks that make model capabilities easier to measure, compare, and reproduce.
  </p>
</section>

{% if site.data.blogs and site.data.blogs != empty %}
<section class="blog-section" id="blog" aria-labelledby="blog-title">
  <div class="section-title-row">
    <h2 class="section-heading" id="blog-title">Latest Writing</h2>
    <a class="section-more" href="{{ '/blog/' | relative_url }}">All Posts →</a>
  </div>
  <div class="blog-list blog-list-compact">
    {% for post in site.data.blogs limit: 3 %}
      <article class="blog-card">
        <time datetime="{{ post.date }}">{{ post.date }}</time>
        <div>
          <h3>{% if post.url %}<a href="{{ post.url }}">{{ post.title }}</a>{% else %}{{ post.title }}{% endif %}</h3>
          {% if post.summary %}<p>{{ post.summary }}</p>{% endif %}
        </div>
      </article>
    {% endfor %}
  </div>
</section>
{% endif %}

<section class="news-section" id="news" aria-labelledby="news-title">
  <div class="section-title-row">
    <h2 class="section-heading" id="news-title">Latest News</h2>
    <a class="section-more" href="{{ '/news/' | relative_url }}">All News →</a>
  </div>
  <ul class="news-list-v2">
    {% assign featured_news = site.data.news | where: "featured", true %}
    {% for item in featured_news limit: 9 %}
      <li>
        <time datetime="{{ item.date }}">{{ item.label }}</time>
        <p>{{ item.content }}</p>
      </li>
    {% endfor %}
  </ul>
</section>

<section class="works-section" id="publications" aria-labelledby="works-title">
  <div class="section-title-row">
    <h2 class="section-heading" id="works-title">Selected Works</h2>
    <a class="section-more" href="{{ '/publications/' | relative_url }}">All Publications →</a>
  </div>
  <div class="works-legend" aria-label="Author contribution legend">
    <span>First / Co-First Author</span>
    <span>Corresponding Author</span>
    <span>Project Lead</span>
  </div>

  {% assign selected_works = site.data.profile.publications | where: "selected", true %}
  <div class="works-list">
    {% for paper in selected_works %}
      {% assign primary_url = paper.links.paper | default: paper.links.scholar %}
      <article class="work-card">
        <a class="paper-visual visual-tone-{{ paper.visual_tone }}{% if paper.thumbnail != empty %} has-thumbnail{% endif %}" href="{{ primary_url }}" aria-label="Open {{ paper.title }}">
          {% if paper.thumbnail != empty %}<img src="{{ paper.thumbnail | relative_url }}" alt="Figure from {{ paper.short_title }}" loading="lazy" decoding="async">{% endif %}
          <span class="paper-venue">{{ paper.venue_short }}</span>
          <span class="paper-visual-caption">
            <strong>{{ paper.short_title }}</strong>
            <small>{{ paper.topics | join: " · " }}</small>
          </span>
        </a>
        <div class="work-copy">
          <h3 class="work-title"><a href="{{ primary_url }}">{{ paper.title }}</a></h3>
          {% if paper.summary != empty %}<p class="work-summary">{{ paper.summary }}</p>{% endif %}
          <div class="work-tags" aria-label="Research topics">
            {% for topic in paper.topics %}<span>{{ topic }}</span>{% endfor %}
          </div>
          <p class="work-authors">
            {% include publication-authors.html authors=paper.authors %}
          </p>
          <p class="work-venue">{{ paper.venue_display }}</p>
          <div class="work-metadata">
            <a href="{{ paper.links.scholar }}" aria-label="View {{ paper.title }} on Google Scholar">Cited by {{ paper.citations }}</a>
            {% if paper.github.url %}<a href="{{ paper.github.url }}" aria-label="View {{ paper.short_title }} on GitHub">★ {{ paper.github.stars_display }} GitHub Stars{% if paper.repo_label != empty %} · {{ paper.repo_label }}{% endif %}</a>{% endif %}
          </div>
        </div>
      </article>
    {% endfor %}
  </div>
</section>

<section class="projects-section" id="projects" aria-labelledby="projects-title">
  <h2 class="section-heading" id="projects-title">Open-Source Projects</h2>
  <div class="project-grid">
    {% for project in site.data.profile.projects %}
      <article class="project-card">
        <div class="project-card-heading">
          <h3><a href="{{ project.url }}">{{ project.name }}</a></h3>
          {% if project.language %}<span class="project-language">{{ project.language }}</span>{% endif %}
        </div>
        <p>{{ project.description }}</p>
        <div class="project-topics" aria-label="Research topics">
          {% for topic in project.topics %}<span>{{ topic }}</span>{% endfor %}
        </div>
        <div class="project-footer">
          <div class="project-stats" aria-label="GitHub statistics">
            <span title="GitHub Stars">★ {{ project.stars_display }}</span>
            <span title="GitHub Forks">⑂ {{ project.forks_display }}</span>
          </div>
          <a href="{{ project.url }}">GitHub →</a>
        </div>
      </article>
    {% endfor %}
  </div>
</section>
