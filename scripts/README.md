# Profile data synchronization

Run the following command from the repository root:

```bash
python3 scripts/update_profile_data.py
```

The script updates `_data/profile.json` from:

- Haodong Duan's Google Scholar profile: all paginated publications, citation counts, and citation rank.
- Every Scholar publication detail page: full author list, publication date, journal/conference/book, volume, issue, pages, publisher, institution, report/patent metadata, abstract/description, and source paper URL when Scholar provides them. Unknown bibliographic fields are preserved instead of discarded.
- GitHub: current Star, Fork, watcher, language, and repository metadata for the configured projects and publication repositories.
- `_publications/*.md`: existing paper/code/project links and legacy contribution markers.
- `scripts/profile_overrides.json`: curated topics, exact paper-verified author contributions, one-line summaries, repository mappings, thumbnails, project list, and Selected Works policy.
- `resume/resume.tex`: explicit `Corresponding Author` and `Project Lead` markers for Haodong, synchronized from the maintained Overleaf CV.

On the first run, the script opens each Scholar publication detail page to obtain complete authors and bibliographic metadata. Later runs reuse those details and normally update the full Scholar publication index, citation counts, and GitHub statistics. Use `--refresh-details` to re-crawl all detail pages when their metadata has changed:

```bash
python3 scripts/update_profile_data.py --refresh-details
```

The generated `sync` object records how many profile rows and unique Scholar IDs were seen, how many detail pages were fetched or failed, and the author/metadata coverage. The Publications page surfaces those coverage counts so an incomplete refresh is visible instead of silently passing.

GitHub's unauthenticated API is rate-limited. The script automatically falls back to public repository HTML for Star/Fork counts. Supplying a token gives richer metadata:

```bash
GITHUB_TOKEN=... python3 scripts/update_profile_data.py
```

Google Scholar does not provide a reliable machine-readable source for equal-contribution, corresponding-author, or project-lead roles. Keep paper-verified annotations in `scripts/profile_overrides.json`; for an overridden paper the role map is treated as the exact curated source and is displayed with colored dots instead of legacy `*`, `†`, or `‡` symbols. The synchronizer also imports Haodong's explicit `\\dag`/`\\ddag` markers from `resume/resume.tex`. When no author has an explicit corresponding-author mark, the project convention assigns that role to the final author and records the fallback in `corresponding_fallback_author`. Strictly alphabetical technical-report lists and institution-authored reports are automatically left entirely unannotated; use `"author_order": "alphabetical"` or `"suppress_author_roles": true` for ambiguous cases.

Per-paper metadata can be updated ahead of Scholar with `venue`, `venue_display`, `venue_short`, and `year` fields in `scripts/profile_overrides.json`. Prefer an official proceedings page, conference accepted-paper list, or publisher record when applying these overrides.

Selected Works is generated using three configurable rules:

1. First/equal/corresponding-author or project-lead papers with at least `lead_min_citations` citations.
2. Papers in the top `top_citation_rank` by citations with no more than `top_cited_max_authors` authors.
3. Per-paper `force_include` and `force_exclude` overrides.

The shared homepage/All News timeline lives in `_data/news.json`. Add a dated item there, mark only the most important items with `"featured": true`, and the homepage will remain a concise subset while `/news/` shows the complete timeline.

## Writing a blog post

Add a Markdown (`.md`) or HTML (`.html`) file under `_blog/`. Both formats use the same article layout, and automatically appear in the homepage's Blog section (latest three), the Blog navigation, and the `/blog/` archive, newest first. No separate index entry or build plugin is needed. The original template posts in `_posts/` are excluded from this Blog index.

For example, `_blog/my-first-note.md` becomes `/blog/my-first-note/`:

```markdown
---
title: "My first note"
date: 2026-11-03
lang: zh-CN
description: "A short search and social preview description."
cover: /images/blog/hello-world.svg
cover_alt: "Describe the cover image."
---

## A section heading

Write the article in **Markdown**.
```

Only `title` and `date` are required; `date` uses `YYYY-MM-DD`. Cover and language are optional. Cards display only the cover and title. Article pages display the title, optional cover, and the Markdown or HTML body. An empty body creates a cover-only post, as in Hello World. Use `description` for a search/social summary and `permalink` to override the URL. Use `published: false` to withhold an article from the normal build and index.

For an HTML article, use the same metadata header in a `.html` file, followed by an HTML body fragment:

```html
---
title: "An HTML note"
date: 2026-11-03
description: "An article with custom HTML content."
---

<h2>A section heading</h2>
<p>Write the article using <strong>HTML</strong>.</p>
```

Do not add a second `<html>`, `<head>`, or `<body>` wrapper; the shared layout provides those. Markdown can also contain inline HTML. The article stylesheet covers headings, code, quotations, lists, images, and tables at desktop and mobile widths.

`_data/blogs.json` remains available for optional external articles. Each entry accepts `date`, `title`, `url`, and an optional `cover` (other metadata is retained but not displayed on cards). External links and local articles share the same date-sorted index. An empty index hides Blog from the homepage and navigation.
