---
layout: profile
title: "Publications"
permalink: /publications/
author_profile: false
---

<section class="publication-index" aria-labelledby="publication-index-title">
  <header class="publication-index-header">
    <p class="publication-eyebrow">GOOGLE SCHOLAR · AUTO-SYNCED</p>
    <h1 id="publication-index-title">Publications</h1>
    <p class="publication-introduction">
      All {{ site.data.profile.profile.publication_count }} records currently listed on
      <a href="{{ site.data.profile.sources.scholar }}">Google Scholar</a> are included.
      Author annotations are curated separately from research-topic tags. When no corresponding author is explicitly identified, the final author is marked as corresponding; alphabetical and team-authored reports are intentionally left unannotated.
    </p>
    <div class="publication-metrics" aria-label="Google Scholar metrics">
      <span><strong>{{ site.data.profile.profile.publication_count }}</strong> publications</span>
      <span><strong>{{ site.data.profile.profile.metrics_display.citations }}</strong> citations</span>
      <span><strong>{{ site.data.profile.profile.metrics.h_index }}</strong> h-index</span>
      <span><strong>{{ site.data.profile.profile.metrics.i10_index }}</strong> i10-index</span>
    </div>
    <div class="works-legend publication-legend" aria-label="Author contribution legend">
      <span>First / Co-First Author</span>
      <span>Corresponding Author</span>
      <span>Project Lead</span>
    </div>
    <p class="publication-coverage">
      {{ site.data.profile.profile.metadata_count }}/{{ site.data.profile.profile.publication_count }} detail records captured
      <span aria-hidden="true">·</span>
      {{ site.data.profile.profile.full_author_count }}/{{ site.data.profile.profile.publication_count }} complete author lists
    </p>
    <p class="publication-updated">Last synchronized {{ site.data.profile.generated_at | date: "%Y-%m-%d %H:%M UTC" }}.</p>
  </header>

  {% assign publication_years = site.data.profile.publications | group_by: "year" %}
  <div class="publication-controls" data-publication-controls>
    <div class="publication-control-head">
      <div class="publication-mode-switch" role="group" aria-label="Browse publications by year or research topic">
        <button class="is-active" type="button" data-publication-mode="year" aria-pressed="true">By year</button>
        <button type="button" data-publication-mode="topic" aria-pressed="false">By topic</button>
      </div>
      <label class="publication-sort">
        <span>Sort by year</span>
        <select data-publication-sort>
          <option value="desc">Newest first</option>
          <option value="asc">Oldest first</option>
        </select>
      </label>
    </div>

    <div class="publication-filter-panel" data-publication-panel="year">
      <span class="publication-filter-label">Select year</span>
      <div class="publication-filter-options" role="group" aria-label="Filter publications by year">
        <button class="is-active" type="button" data-filter-kind="year" data-filter-value="all" aria-pressed="true">All</button>
        {% for year in publication_years %}<button type="button" data-filter-kind="year" data-filter-value="{{ year.name }}" aria-pressed="false">{{ year.name }}</button>{% endfor %}
      </div>
    </div>

    <div class="publication-filter-panel" data-publication-panel="topic" hidden>
      <span class="publication-filter-label">Select topic</span>
      <div class="publication-filter-options publication-topic-options" role="group" aria-label="Filter publications by research topic" data-publication-topics>
        <button class="is-active" type="button" data-filter-kind="topic" data-filter-value="all" aria-pressed="true">All</button>
      </div>
    </div>

    <div class="publication-filter-status" aria-live="polite">
      <span><strong data-publication-visible-count>{{ site.data.profile.profile.publication_count }}</strong> of {{ site.data.profile.profile.publication_count }} publications</span>
      <span data-publication-selection>All years</span>
      <button type="button" data-publication-reset hidden>Reset filters</button>
    </div>
  </div>

  <div class="publication-list" data-publication-list>
    {% for year in publication_years %}
      <section class="publication-year-group" data-year-group="{{ year.name }}">
        <h2 class="publication-year">{{ year.name }}</h2>
        {% for paper in year.items %}
          {% assign primary_url = paper.links.paper | default: paper.links.code %}
          <article class="publication-row" data-publication-row data-year="{{ paper.year }}" data-topics="{{ paper.topics | join: '||' | escape }}">
            <div class="publication-row-main">
              <h3 class="publication-row-title">{% if primary_url %}<a href="{{ primary_url }}">{{ paper.title }}</a>{% else %}{{ paper.title }}{% endif %}</h3>
              <div class="work-tags" aria-label="Research topics">
                {% for topic in paper.topics %}<a class="publication-topic-link" href="{{ '/publications/' | relative_url }}?tag={{ topic | url_encode }}">{{ topic }}</a>{% endfor %}
              </div>
              <p class="work-authors">
                {% include publication-authors.html authors=paper.authors %}
              </p>
              <p class="work-venue">{{ paper.venue_display }}</p>
              {% if paper.metadata.size > 0 or paper.description != empty %}
                <details class="publication-metadata">
                  <summary>Full metadata</summary>
                  {% if paper.metadata.size > 0 %}
                    <dl>
                      {% for item in paper.metadata %}
                        <div>
                          <dt>{{ item.label }}</dt>
                          <dd>{{ item.value }}</dd>
                        </div>
                      {% endfor %}
                    </dl>
                  {% endif %}
                  {% if paper.description != empty %}
                    <div class="publication-description">
                      <h4>Description</h4>
                      <p>{{ paper.description }}</p>
                    </div>
                  {% endif %}
                </details>
              {% endif %}
              <div class="work-links">
                {% if paper.links.paper %}<a href="{{ paper.links.paper }}">Paper</a>{% endif %}
                {% if paper.links.code %}<a href="{{ paper.links.code }}">Code</a>{% endif %}
              </div>
            </div>
            <div class="publication-row-stats">
              <strong>{{ paper.citations }}</strong>
              <span>citations</span>
            </div>
          </article>
        {% endfor %}
      </section>
    {% endfor %}
  </div>

  <p class="publication-empty" data-publication-empty hidden>No publications match this filter.</p>
</section>

<script>
  (function () {
    var controls = document.querySelector('[data-publication-controls]');
    var list = document.querySelector('[data-publication-list]');
    if (!controls || !list) return;

    var rows = Array.prototype.slice.call(list.querySelectorAll('[data-publication-row]'));
    var groups = Array.prototype.slice.call(list.querySelectorAll('[data-year-group]'));
    var modeButtons = Array.prototype.slice.call(controls.querySelectorAll('[data-publication-mode]'));
    var panels = Array.prototype.slice.call(controls.querySelectorAll('[data-publication-panel]'));
    var topicsContainer = controls.querySelector('[data-publication-topics]');
    var sortSelect = controls.querySelector('[data-publication-sort]');
    var visibleCount = controls.querySelector('[data-publication-visible-count]');
    var selection = controls.querySelector('[data-publication-selection]');
    var resetButton = controls.querySelector('[data-publication-reset]');
    var emptyState = document.querySelector('[data-publication-empty]');
    var state = { mode: 'year', year: 'all', topic: 'all', sort: 'desc' };

    var topicMap = {};
    rows.forEach(function (row) {
      (row.getAttribute('data-topics') || '').split('||').forEach(function (topic) {
        if (topic) topicMap[topic] = (topicMap[topic] || 0) + 1;
      });
    });

    Object.keys(topicMap).sort(function (a, b) {
      return topicMap[b] - topicMap[a] || a.localeCompare(b, 'en', { sensitivity: 'base' });
    }).forEach(function (topic) {
      var button = document.createElement('button');
      button.type = 'button';
      button.setAttribute('data-filter-kind', 'topic');
      button.setAttribute('data-filter-value', topic);
      button.setAttribute('aria-pressed', 'false');
      button.textContent = topic + ' (' + topicMap[topic] + ')';
      topicsContainer.appendChild(button);
    });

    function filterButtons(kind) {
      return Array.prototype.slice.call(controls.querySelectorAll('[data-filter-kind="' + kind + '"]'));
    }

    function syncButtons() {
      modeButtons.forEach(function (button) {
        var active = button.getAttribute('data-publication-mode') === state.mode;
        button.classList.toggle('is-active', active);
        button.setAttribute('aria-pressed', active ? 'true' : 'false');
      });
      panels.forEach(function (panel) {
        panel.hidden = panel.getAttribute('data-publication-panel') !== state.mode;
      });
      ['year', 'topic'].forEach(function (kind) {
        filterButtons(kind).forEach(function (button) {
          var active = button.getAttribute('data-filter-value') === state[kind];
          button.classList.toggle('is-active', active);
          button.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
      });
      sortSelect.value = state.sort;
    }

    function writeUrl() {
      if (!window.history || !window.URLSearchParams) return;
      var params = new URLSearchParams();
      if (state.year !== 'all') params.set('year', state.year);
      if (state.topic !== 'all') params.set('tag', state.topic);
      if (state.sort === 'asc') params.set('sort', 'oldest');
      var query = params.toString();
      window.history.replaceState(null, '', window.location.pathname + (query ? '?' + query : '') + window.location.hash);
    }

    function applyFilters(updateUrl) {
      groups.sort(function (a, b) {
        var first = Number(a.getAttribute('data-year-group'));
        var second = Number(b.getAttribute('data-year-group'));
        return state.sort === 'asc' ? first - second : second - first;
      }).forEach(function (group) {
        list.appendChild(group);
      });

      var count = 0;
      groups.forEach(function (group) {
        var groupCount = 0;
        Array.prototype.slice.call(group.querySelectorAll('[data-publication-row]')).forEach(function (row) {
          var topics = (row.getAttribute('data-topics') || '').split('||');
          var matchesYear = state.year === 'all' || row.getAttribute('data-year') === state.year;
          var matchesTopic = state.topic === 'all' || topics.indexOf(state.topic) !== -1;
          var show = matchesYear && matchesTopic;
          row.hidden = !show;
          if (show) {
            groupCount += 1;
            count += 1;
          }
        });
        group.hidden = groupCount === 0;
      });

      visibleCount.textContent = count;
      if (state.topic !== 'all') {
        selection.textContent = 'Topic: ' + state.topic;
      } else if (state.year !== 'all') {
        selection.textContent = 'Year: ' + state.year;
      } else {
        selection.textContent = 'All publications';
      }
      resetButton.hidden = state.year === 'all' && state.topic === 'all' && state.sort === 'desc';
      emptyState.hidden = count !== 0;
      syncButtons();
      if (updateUrl) writeUrl();
    }

    modeButtons.forEach(function (button) {
      button.addEventListener('click', function () {
        state.mode = button.getAttribute('data-publication-mode');
        if (state.mode === 'year') state.topic = 'all';
        if (state.mode === 'topic') state.year = 'all';
        applyFilters(true);
      });
    });

    controls.addEventListener('click', function (event) {
      var button = event.target.closest('[data-filter-kind]');
      if (!button || !controls.contains(button)) return;
      var kind = button.getAttribute('data-filter-kind');
      state[kind] = button.getAttribute('data-filter-value');
      applyFilters(true);
    });

    sortSelect.addEventListener('change', function () {
      state.sort = sortSelect.value;
      applyFilters(true);
    });

    resetButton.addEventListener('click', function () {
      state = { mode: 'year', year: 'all', topic: 'all', sort: 'desc' };
      applyFilters(true);
    });

    if (window.URLSearchParams) {
      var params = new URLSearchParams(window.location.search);
      var requestedTag = params.get('tag');
      var requestedYear = params.get('year');
      if (requestedTag && topicMap[requestedTag]) {
        state.mode = 'topic';
        state.topic = requestedTag;
      } else if (requestedYear && groups.some(function (group) { return group.getAttribute('data-year-group') === requestedYear; })) {
        state.mode = 'year';
        state.year = requestedYear;
      }
      if (params.get('sort') === 'oldest') state.sort = 'asc';
    }

    applyFilters(false);
  }());
</script>
