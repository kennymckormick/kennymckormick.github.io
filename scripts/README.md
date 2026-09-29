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

Blog data lives in `_data/blogs.json`. It is intentionally an empty JSON array until the first post exists. Each entry accepts `date`, `title`, `url`, `summary`, and an optional `tags` array. Once at least one entry is present, the Blog navigation item, homepage `Latest Writing` section, and `/blog/` archive appear automatically.
