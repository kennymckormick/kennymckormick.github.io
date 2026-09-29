#!/usr/bin/env python3
"""Synchronize Google Scholar publications and GitHub project statistics.

The generated `_data/profile.json` is consumed directly by Jekyll. Google
Scholar does not expose reliable contribution-role metadata, so automatic
publication data is merged with the curated annotations in
`scripts/profile_overrides.json` and with legacy `_publications/*.md` metadata.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import difflib
import html
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OVERRIDES = ROOT / "scripts" / "profile_overrides.json"
DEFAULT_OUTPUT = ROOT / "_data" / "profile.json"
DEFAULT_RESUME_SOURCE = ROOT / "resume" / "resume.tex"
SCHOLAR_BASE = "https://scholar.google.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140 Safari/537.36"
)
ROLE_ORDER = ("first", "equal", "corresponding", "project_lead")
NON_BIBLIOGRAPHIC_DETAIL_FIELDS = {
    "authors",
    "description",
    "total citations",
    "scholar articles",
}
DETAIL_FIELD_LABELS = {
    "publication date": "Publication date",
    "journal": "Journal",
    "conference": "Conference",
    "book": "Book",
    "volume": "Volume",
    "issue": "Issue",
    "pages": "Pages",
    "publisher": "Publisher",
    "institution": "Institution",
    "report number": "Report number",
    "patent number": "Patent number",
    "application number": "Application number",
    "inventors": "Inventors",
}


class FetchError(RuntimeError):
    pass


class Node:
    def __init__(self, tag: str, attrs: dict[str, str], parent: "Node | None" = None):
        self.tag = tag
        self.attrs = attrs
        self.parent = parent
        self.children: list[Node | str] = []


class DOMBuilder(HTMLParser):
    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document", {})
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag, {key: value or "" for key, value in attrs}, self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in self.VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if self.stack[-1].tag == tag:
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def parse_dom(source: str) -> Node:
    parser = DOMBuilder()
    parser.feed(source)
    return parser.root


def class_names(node: Node) -> set[str]:
    return set(node.attrs.get("class", "").split())


def walk(node: Node) -> Iterable[Node]:
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(child for child in reversed(current.children) if isinstance(child, Node))


def find_all(node: Node, *, tag: str | None = None, cls: str | None = None,
             node_id: str | None = None) -> list[Node]:
    matches = []
    for candidate in walk(node):
        if tag and candidate.tag != tag:
            continue
        if cls and cls not in class_names(candidate):
            continue
        if node_id and candidate.attrs.get("id") != node_id:
            continue
        matches.append(candidate)
    return matches


def first(node: Node, *, tag: str | None = None, cls: str | None = None,
          node_id: str | None = None) -> Node | None:
    found = find_all(node, tag=tag, cls=cls, node_id=node_id)
    return found[0] if found else None


def node_text(node: Node | None, skip_classes: set[str] | None = None) -> str:
    if node is None:
        return ""
    skip_classes = skip_classes or set()
    pieces: list[str] = []

    def collect(item: Node | str) -> None:
        if isinstance(item, str):
            pieces.append(item)
            return
        if item.tag in {"script", "style"} or class_names(item) & skip_classes:
            return
        for child in item.children:
            collect(child)

    collect(node)
    return normalize_space("".join(pieces))


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    return re.sub(r"[^a-z0-9]+", "", value)


def normalize_name(value: str) -> str:
    value = clean_author_name(value)
    value = unicodedata.normalize("NFKD", value).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def clean_author_name(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", html.unescape(value or ""))
    value = value.replace("\u2026", "...")
    value = re.sub(r"[\*\u2217\u2020\u2021]+", "", value)
    return normalize_space(value).strip(" ,")


def parse_int(value: str | int | None) -> int:
    if isinstance(value, int):
        return value
    match = re.search(r"\d[\d,]*", value or "")
    return int(match.group(0).replace(",", "")) if match else 0


def http_get(url: str, *, headers: dict[str, str] | None = None,
             timeout: float = 30.0) -> str:
    request_headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    }
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            return body.decode(charset, errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise FetchError(f"Unable to fetch {url}: {exc}") from exc


def profile_url(scholar_id: str, cstart: int = 0) -> str:
    parameters: dict[str, Any] = {"user": scholar_id, "hl": "en", "pagesize": 100}
    if cstart:
        parameters["cstart"] = cstart
    query = urllib.parse.urlencode(parameters)
    return f"{SCHOLAR_BASE}/citations?{query}"


def parse_scholar_profile(source: str, scholar_id: str) -> dict[str, Any]:
    if "gsc_a_tr" not in source or "Google Scholar" not in source:
        raise FetchError("Google Scholar returned an unexpected or blocked response")
    root = parse_dom(source)
    publications: list[dict[str, Any]] = []
    for row in find_all(root, tag="tr", cls="gsc_a_tr"):
        title_node = first(row, tag="a", cls="gsc_a_at")
        if not title_node:
            continue
        gray_nodes = find_all(row, tag="div", cls="gs_gray")
        citation_node = first(row, tag="a", cls="gsc_a_ac")
        year_node = first(row, tag="td", cls="gsc_a_y")
        detail_href = title_node.attrs.get("href", "")
        detail_url = urllib.parse.urljoin(SCHOLAR_BASE, detail_href)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(detail_url).query)
        publication_id = query.get("citation_for_view", [""])[0]
        citation_url = citation_node.attrs.get("href", "") if citation_node else ""
        venue = node_text(gray_nodes[1], {"gs_oph"}) if len(gray_nodes) > 1 else ""
        venue_years = re.findall(r"\b(?:19|20)\d{2}\b", venue)
        publications.append({
            "scholar_id": publication_id,
            "title": node_text(title_node),
            "authors_raw": node_text(gray_nodes[0]) if gray_nodes else "",
            "venue": venue,
            "year": int(venue_years[-1]) if venue_years else parse_int(node_text(year_node)),
            "citations": parse_int(node_text(citation_node)),
            "detail_url": detail_url,
            "citation_url": urllib.parse.urljoin(SCHOLAR_BASE, citation_url),
        })

    metric_values = [parse_int(node_text(node)) for node in find_all(root, tag="td", cls="gsc_rsb_std")]
    profile_name = node_text(first(root, node_id="gsc_prf_in"))
    interests = [node_text(node) for node in find_all(root, tag="a", cls="gsc_prf_inta")]
    description = ""
    for node in find_all(root, tag="meta"):
        if node.attrs.get("name") == "description":
            description = node.attrs.get("content", "")
            break
    description_citations = re.search(r"Cited by\s*([\d,]+)", description)
    metrics = {
        "citations": metric_values[0] if metric_values else (
            parse_int(description_citations.group(1)) if description_citations else 0
        ),
        "citations_recent": metric_values[1] if len(metric_values) > 1 else None,
        "h_index": metric_values[2] if len(metric_values) > 2 else None,
        "h_index_recent": metric_values[3] if len(metric_values) > 3 else None,
        "i10_index": metric_values[4] if len(metric_values) > 4 else None,
        "i10_index_recent": metric_values[5] if len(metric_values) > 5 else None,
    }
    return {
        "name": profile_name,
        "interests": interests,
        "metrics": metrics,
        "publications": publications,
    }


def next_sibling_with_class(node: Node, cls: str) -> Node | None:
    if not node.parent:
        return None
    seen = False
    for sibling in node.parent.children:
        if sibling is node:
            seen = True
            continue
        if seen and isinstance(sibling, Node) and cls in class_names(sibling):
            return sibling
    return None


def parse_scholar_detail(source: str) -> dict[str, Any]:
    if "gsc_oci_title" not in source:
        raise FetchError("Google Scholar detail page was blocked or incomplete")
    root = parse_dom(source)
    fields: dict[str, str] = {}
    for field_node in find_all(root, tag="div", cls="gsc_oci_field"):
        value_node = next_sibling_with_class(field_node, "gsc_oci_value")
        if value_node:
            fields[node_text(field_node).casefold()] = node_text(value_node)
    title_link = first(root, tag="a", cls="gsc_oci_title_link")
    return {
        "title": node_text(first(root, node_id="gsc_oci_title")),
        "paper_url": title_link.attrs.get("href", "") if title_link else "",
        "authors_raw": fields.get("authors", ""),
        "fields": fields,
    }


def metadata_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")


def scholar_metadata(detail: dict[str, Any] | None,
                     previous: dict[str, Any] | None) -> tuple[list[dict[str, str]], str]:
    """Return every bibliographic field exposed by a Scholar detail page.

    Authors, description, and citation widgets are rendered through dedicated
    fields elsewhere. Unknown bibliographic labels are retained rather than
    discarded so the synchronizer remains forward-compatible with Scholar.
    """
    if detail is None:
        return (
            list((previous or {}).get("metadata", [])),
            str((previous or {}).get("description", "")),
        )

    fields = detail.get("fields", {})
    metadata: list[dict[str, str]] = []
    for raw_label, raw_value in fields.items():
        label = normalize_space(str(raw_label)).casefold()
        value = normalize_space(str(raw_value))
        if not value or label in NON_BIBLIOGRAPHIC_DETAIL_FIELDS:
            continue
        metadata.append({
            "key": metadata_key(label),
            "label": DETAIL_FIELD_LABELS.get(label, label[:1].upper() + label[1:]),
            "value": value,
        })
    return metadata, normalize_space(str(fields.get("description", "")))


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value[0:1] in {"'", '"'}:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value.strip("'\"")
    if value.isdigit():
        return int(value)
    if value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    return value


def read_front_matter(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    result: dict[str, Any] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = re.match(r"^([A-Za-z0-9_]+):\s*(.*)$", line)
        if match:
            result[match.group(1)] = parse_scalar(match.group(2))
    return result


def parse_authors(raw: str, self_aliases: set[str]) -> tuple[list[dict[str, Any]], bool]:
    parts = [part.strip() for part in re.split(r"\s*,\s*", raw or "")]
    authors: list[dict[str, Any]] = []
    truncated = False
    for index, part in enumerate(parts):
        if not part:
            continue
        if part in {"...", "…"} or re.fullmatch(r"et\s+al\.?", part, re.I):
            truncated = True
            continue
        roles: list[str] = []
        if "*" in part or "∗" in part:
            roles.append("equal")
        if "†" in part:
            roles.append("corresponding")
        if "‡" in part:
            roles.append("project_lead")
        name = clean_author_name(part)
        if not name:
            continue
        if index == 0:
            roles.append("first")
        normalized = normalize_name(name)
        authors.append({
            "name": name,
            "is_self": normalized in self_aliases,
            "roles": sorted(set(roles), key=ROLE_ORDER.index),
        })
    return authors, truncated


def legacy_publications(self_aliases: set[str]) -> dict[str, dict[str, Any]]:
    publications: dict[str, dict[str, Any]] = {}
    for path in sorted((ROOT / "_publications").glob("*.md")):
        data = read_front_matter(path)
        title = str(data.get("title", ""))
        if not title:
            continue
        authors, truncated = parse_authors(str(data.get("author", "")), self_aliases)
        publications[normalize_title(title)] = {
            **data,
            "authors": authors,
            "authors_truncated": truncated,
            "source_file": str(path.relative_to(ROOT)),
        }
    return publications


def match_legacy(title: str, legacy: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    normalized = normalize_title(title)
    if normalized in legacy:
        return legacy[normalized]
    matches = difflib.get_close_matches(normalized, legacy.keys(), n=1, cutoff=0.94)
    return legacy[matches[0]] if matches else None


def roles_by_name(authors: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {normalize_name(author["name"]): set(author.get("roles", [])) for author in authors}


def surname_key(name: str) -> str:
    """Return a conservative surname key for romanized author lists."""
    pieces = normalize_name(name).split()
    return pieces[-1] if pieces else ""


def is_alphabetical_report(title: str, venue: str, authors: list[dict[str, Any]]) -> bool:
    """Detect report-style author lists that are ordered by surname.

    Exact ordering is convincing for a long consortium-style list even when
    its title omits "technical report". Report/model-card lists may contain a
    late metadata insertion, so they can tolerate one or two local inversions.
    Ordinary papers, surveys, and short lists retain contribution annotations.
    """
    if len(authors) < 5:
        return False
    keys = [surname_key(author["name"]) for author in authors]
    if not all(keys) or len(set(keys)) < 3:
        return False
    if len(authors) >= 20 and keys == sorted(keys):
        return True
    report_like = bool(re.search(r"\breport\b|\bmodel card\b", f"{title} {venue}", re.I))
    ordered_pairs = sum(left <= right for left, right in zip(keys, keys[1:]))
    ordered_ratio = ordered_pairs / max(len(keys) - 1, 1)
    return report_like and ordered_ratio >= 0.95


def clean_tex_title(value: str) -> str:
    value = value.replace(r"\&", "&").replace("~", " ")
    value = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\[A-Za-z]+", " ", value)
    return normalize_space(value)


def load_resume_self_roles(path: Path) -> dict[str, set[str]]:
    """Read Haodong's explicit role markers from the maintained LaTeX CV.

    The CV uses ``\\dag`` for corresponding author and ``\\ddag`` for project
    lead. It is supplemental evidence only; paper-level overrides remain the
    authoritative source for every author's complete role set.
    """
    if not path.exists():
        return {}
    source = path.read_text(encoding="utf-8")
    entry_pattern = re.compile(
        r"\\textbf\{\[[^}]+\]\}\s*\\textbf\{(?P<title>.*?)\}"
        r"(?P<body>.*?)(?=\\itemgap|\\texttt\{|\\end\{document\})",
        re.S,
    )
    result: dict[str, set[str]] = {}
    for match in entry_pattern.finditer(source):
        title = clean_tex_title(match.group("title"))
        body = match.group("body")
        self_match = re.search(r"Haodong\s+Duan(?P<marks>[^,}\n]*)", body, re.I)
        if not title or not self_match:
            continue
        marks = self_match.group("marks")
        roles: set[str] = set()
        if r"\ddag" in marks:
            roles.add("project_lead")
        elif r"\dag" in marks:
            roles.add("corresponding")
        if roles:
            result.setdefault(normalize_title(title), set()).update(roles)
    return result


def is_scholar_url(url: str) -> bool:
    host = urllib.parse.urlparse(url or "").netloc.casefold()
    return host == "scholar.google.com" or host.endswith(".scholar.google.com")


def find_author(authors: list[dict[str, Any]], target: str, self_aliases: set[str]) -> dict[str, Any] | None:
    normalized = normalize_name(target)
    if normalized in self_aliases:
        for author in authors:
            if author.get("is_self"):
                return author
    for author in authors:
        if normalize_name(author["name"]) == normalized:
            return author
    return None


TOPIC_RULES: tuple[tuple[str, str], ...] = (
    ("Medical AI", r"medical|medicine|clinical|gmai|imaging-x"),
    ("Skeleton Recognition", r"skeleton|posec3d|stgcn|skeletr|pyskl"),
    ("Action Recognition", r"action recognition|video recognition|skeleton|pyskl|mmaction|omni-sourced|omnisource"),
    ("Agentic Systems", r"\bagent(?:ic|s)?\b|\bagency\b|computer-use|\bgui\b|workflow|tool use|non-markov|interactive environment|automated research|seed1\.8|seed2\.[01]"),
    ("Long-Context Modeling", r"long[- ]context|long-form|long-term streaming|retrieval head|needlebench|information densit|128k|4khd|contextual input|long video|rotary position embedding"),
    ("Spatial Reasoning", r"spatial|geometric|geometry|affordance|\b3d\b|optics|lego"),
    ("Multi-Image Reasoning", r"multi-image|multiple images|image pairs|paired images"),
    ("Document Intelligence", r"visual document|\bdocument\b|chart|poster|4khd|patch-level embedding"),
    ("Scientific Reasoning", r"scientific|\bscience\b|mathemat|search spaces optimization|optimization problems?|\bnp\b|physics|research system|research-grade|theorem|prover"),
    ("Reward Modeling", r"process reward|reward model|verifiable reward|policy and reward|reward co-|rewarding"),
    ("Reinforcement Learning", r"reinforcement|policy optimization|\brft\b|\brl\b|ssrl"),
    ("Model Alignment", r"alignment|preference|\bdpo\b|instruction following|instruction tuning|self-refinement"),
    ("Multimodal Generation", r"\bgenerat(?:ion|ive|ors?|ing|ed)\b|diffusion|image editing|text-image composition|captioning|text-to-video|\bcreation\b|creative|super resolution"),
    ("Self-Supervised Learning", r"self-supervised|webly-supervised|representation learning|pretrain"),
    ("Video Understanding", r"\bvideo|temporal|streaming|motion|action recognition|video recognition"),
    ("Human Pose", r"human pose|human body|contour keypoint|\bpose\b"),
    ("Image Understanding", r"image understanding|visual understanding|image quality|visual cognition|image retrieval|single image|text-image comprehension|visual instruction"),
    ("Visual Reasoning", r"visual reasoning|reasoning-informed visual|visual cognition|perception|vision-language synergy|\barc\b|puzzle|visual dependency|vision and reasoning|perception-reasoning|implicit world rules"),
    ("Evaluation Infrastructure", r"evaluation toolkit|evaluation platform|toolbox|opencompass|vlmevalkit|scievalkit|compassjudger|mmaction2|pyskl"),
    ("Multimodal Evaluation", r"(?=.*(?:bench|evaluat|assess|survey|judge|diagnos|principle))(?=.*(?:multimodal|multi-modal|mllm|\blmm\b|vision-language|visual|image|video))|mmbench|vlmevalkit|visfactor|gobench|mibench|prism"),
    ("LLM Evaluation", r"(?=.*(?:bench|evaluat|assess|survey|judge|sensitivity|retrieval))(?=.*(?:\bllm|language model|scientific intelligence|prompt))|opencompass|mathbench|needlebench|botchat|prosa|atlas|scievalkit|compassjudger"),
    ("Benchmark Design", r"bench|evaluat|assessment|survey|judge|sensitivity|redundancy|information density|data leakage|capability diagnosis|circular"),
    ("Data-Centric Learning", r"dataset|data synthesis|training data|caption data|web data|multi-source|omni-sourced|data foundation|data collection|annotation"),
    ("Model Efficiency", r"efficien|compress|storage|latency|token efficiency|sampling|lightweight|redundancy"),
    ("Model Adaptation", r"fine-tun|instruction tun|\btraining\b|adaptation|\bdpo\b|reward model|\bpolicy\b|alignment"),
    ("Foundation Models", r"foundation model|model card|technical report|large language model|vision-language model|multimodal model|internlm|internvl|xcomposer|seed[12]|\bmllm|\blmm|\bvlm|\bllm"),
)

CANONICAL_TOPICS = frozenset(topic for topic, _ in TOPIC_RULES)

LEGACY_TOPIC_ALIASES = {
    "Evaluation": "Benchmark Design",
    "Large Language Models": "Foundation Models",
    "Vision-Language Models": "Foundation Models",
    "Multimodal Learning": "Foundation Models",
    "Machine Learning": "Foundation Models",
    "Generative AI": "Multimodal Generation",
    "AI Agents": "Agentic Systems",
    "Skeleton Action": "Skeleton Recognition",
    "Capability Diagnosis": "Benchmark Design",
}


def infer_topics(title: str) -> list[str]:
    """Assign a compact, reusable research taxonomy from the paper title."""
    haystack = title.casefold()
    topics = [topic for topic, pattern in TOPIC_RULES if re.search(pattern, haystack, re.I)]
    topics = list(dict.fromkeys(topics))[:3]
    if not topics:
        topics.append("Foundation Models")
    if len(topics) == 1:
        companion = "Model Adaptation" if topics[0] == "Foundation Models" else "Foundation Models"
        topics.append(companion)
    return topics


def canonical_topics(title: str, curated: list[str] | None = None) -> list[str]:
    """Preserve curated canonical tags and fill sparse entries from the shared taxonomy."""
    topics: list[str] = []
    for raw_topic in curated or []:
        topic = LEGACY_TOPIC_ALIASES.get(raw_topic, raw_topic)
        if topic in CANONICAL_TOPICS and topic not in topics:
            topics.append(topic)
    if len(topics) >= 2:
        return topics[:3]
    for topic in infer_topics(title):
        if topic not in topics:
            topics.append(topic)
    return topics[:3]


VENUE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"computer vision and pattern recognition|\bcvpr\b", "CVPR"),
    (r"international conference on computer vision|\biccv\b", "ICCV"),
    (r"european conference on computer vision|\beccv\b", "ECCV"),
    (r"neural information processing systems|\bneurips\b", "NeurIPS"),
    (r"international conference on learning representations|\biclr\b", "ICLR"),
    (r"association for computational linguistics|\bacl\b", "ACL"),
    (r"empirical methods in natural language processing|\bemnlp\b", "EMNLP"),
    (r"north american chapter.*computational linguistics|\bnaacl\b", "NAACL"),
    (r"acm.*multimedia|\bacmmm\b|\bacm mm\b", "ACM MM"),
    (r"aaai", "AAAI"),
    (r"arxiv", "arXiv"),
)


def venue_short(venue: str) -> str:
    for pattern, short in VENUE_PATTERNS:
        if re.search(pattern, venue, re.I):
            return short
    value = re.split(r"[,;(]", venue)[0].strip()
    return value[:18] or "Publication"


def short_title(title: str) -> str:
    lead = re.split(r":| - ", title, maxsplit=1)[0].strip()
    if len(lead) <= 28:
        return lead
    words = lead.split()
    acronym = "".join(word[0] for word in words if word and word[0].isalnum()).upper()
    return acronym if 2 < len(acronym) <= 10 else lead[:27].rstrip() + "…"


def load_previous(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        item.get("scholar_id", ""): item
        for item in data.get("publications", [])
        if item.get("scholar_id")
    }


def merge_publication(
    row: dict[str, Any],
    detail: dict[str, Any] | None,
    previous: dict[str, Any] | None,
    legacy: dict[str, Any] | None,
    override: dict[str, Any],
    self_aliases: set[str],
    resume_self_roles: set[str] | None = None,
) -> dict[str, Any]:
    display_title = str(override.get("title") or (legacy or {}).get("title") or row["title"])
    venue = str(override.get("venue") or (legacy or {}).get("conf") or row.get("venue") or "")
    year = int(override.get("year") or (legacy or {}).get("year") or row.get("year") or 0)
    venue_display = str(override.get("venue_display") or venue)
    if year and not re.search(rf"\b{year}\b", venue_display):
        venue_display = f"{venue_display}, {year}" if venue_display else str(year)
    authors_raw = (detail or {}).get("authors_raw") or row.get("authors_raw", "")
    authors, truncated = parse_authors(authors_raw, self_aliases)
    reused_previous_authors = False
    if detail is None and previous and previous.get("authors") and not previous.get("authors_truncated"):
        authors = [
            {**author, "roles": list(author.get("roles", []))}
            for author in previous["authors"]
        ]
        truncated = False
        reused_previous_authors = True
    elif truncated and previous and previous.get("authors") and not previous.get("authors_truncated"):
        authors = [
            {**author, "roles": list(author.get("roles", []))}
            for author in previous["authors"]
        ]
        truncated = False
        reused_previous_authors = True
    if detail is None and not previous and legacy and legacy.get("authors") and not legacy.get("authors_truncated"):
        authors = legacy["authors"]
        truncated = False
    elif (not authors or truncated) and legacy and legacy.get("authors"):
        authors = legacy["authors"]
        truncated = bool(legacy.get("authors_truncated"))

    # Scholar occasionally exposes an outdated or shortened author list.  A
    # paper-verified list in the overrides is authoritative for those cases.
    if override.get("authors"):
        authors = []
        for index, name in enumerate(override["authors"]):
            authors.append({
                "name": name,
                "is_self": normalize_name(name) in self_aliases,
                "roles": ["first"] if index == 0 else [],
            })
        truncated = False
        reused_previous_authors = False

    # A previous generated record may contain the project-level last-author
    # fallback. Remove that inferred role before re-evaluating current explicit
    # evidence, otherwise a newly curated corresponding author could leave a
    # stale blue annotation on the old fallback author.
    if reused_previous_authors and previous and previous.get("corresponding_fallback_author"):
        previous_fallback = normalize_name(str(previous["corresponding_fallback_author"]))
        for author in authors:
            if normalize_name(author["name"]) == previous_fallback:
                author["roles"] = [
                    role for role in author.get("roles", []) if role != "corresponding"
                ]

    inherited_roles: dict[str, set[str]] = {}
    if legacy:
        inherited_roles = roles_by_name(legacy.get("authors", []))
    for author in authors:
        normalized = normalize_name(author["name"])
        author["roles"] = sorted(
            set(author.get("roles", [])) | inherited_roles.get(normalized, set()),
            key=ROLE_ORDER.index,
        )
        if normalized in self_aliases:
            author["is_self"] = True

    if "author_roles" in override:
        # Curated annotations are an exact snapshot of the contribution marks
        # on the paper, rather than an additive patch over stale front matter.
        for index, author in enumerate(authors):
            author["roles"] = ["first"] if index == 0 else []
        for author_name, roles in override.get("author_roles", {}).items():
            author = find_author(authors, author_name, self_aliases)
            if author:
                author["roles"] = sorted(
                    set(author.get("roles", [])) | set(roles), key=ROLE_ORDER.index
                )

    if "author_roles" not in override and resume_self_roles:
        self_author = find_author(authors, "Haodong Duan", self_aliases)
        if self_author:
            self_author["roles"] = sorted(
                set(self_author.get("roles", [])) | set(resume_self_roles),
                key=ROLE_ORDER.index,
            )

    group_authored = any(
        re.search(r"\b(team|contributors?|collaboration|consortium|seed)\b", author["name"], re.I)
        for author in authors
    )
    author_order_override = str(override.get("author_order") or "").casefold()
    alphabetical_authors = author_order_override == "alphabetical" or (
        author_order_override != "contribution"
        and is_alphabetical_report(display_title, venue, authors)
    )
    author_annotations_suppressed = bool(override.get("suppress_author_roles")) or (
        alphabetical_authors or group_authored
    )
    corresponding_fallback_author = ""
    if author_annotations_suppressed:
        for author in authors:
            author["roles"] = []
    elif authors and not truncated and not any(
        "corresponding" in author.get("roles", []) for author in authors
    ):
        # Project convention requested by Haodong: when the available paper
        # metadata has no explicit corresponding-author mark, annotate the last
        # author as corresponding. Alphabetical/team reports remain unmarked.
        authors[-1]["roles"] = sorted(
            set(authors[-1].get("roles", [])) | {"corresponding"},
            key=ROLE_ORDER.index,
        )
        corresponding_fallback_author = authors[-1]["name"]

    links = dict((previous or {}).get("links", {}))
    links.update({
        "scholar": row.get("detail_url", ""),
        "citations": row.get("citation_url", ""),
    })
    paper_candidates = (
        (legacy or {}).get("paperurl"),
        (detail or {}).get("paper_url"),
        links.get("paper"),
    )
    paper_url = next(
        (str(candidate) for candidate in paper_candidates if candidate and not is_scholar_url(str(candidate))),
        "",
    )
    if paper_url:
        links["paper"] = paper_url
    if legacy and legacy.get("codeurl"):
        links["code"] = str(legacy["codeurl"])
    if legacy and legacy.get("projecturl"):
        links["project"] = str(legacy["projecturl"])
    links.update(override.get("links", {}))
    if is_scholar_url(str(links.get("paper") or "")):
        links.pop("paper", None)

    repo = str(override.get("repo") or (previous or {}).get("repo") or "")
    if repo:
        links["code"] = f"https://github.com/{repo}"

    metadata, description = scholar_metadata(detail, previous)

    summary = normalize_space(str(
        override.get("summary") or (previous or {}).get("summary") or ""
    ))
    if not summary and description:
        summary = re.split(r"(?<=[.!?])\s+", normalize_space(description), maxsplit=1)[0]

    topics = canonical_topics(display_title, override.get("topics"))
    author_count = len(authors) if authors and not truncated else None
    return {
        "scholar_id": row.get("scholar_id", ""),
        "title": display_title,
        "short_title": override.get("short_title") or short_title(display_title),
        "authors": authors,
        "author_count": author_count,
        "authors_truncated": truncated,
        "group_authored": group_authored,
        "author_order": "alphabetical" if alphabetical_authors else ("group" if group_authored else "contribution"),
        "author_annotations_suppressed": author_annotations_suppressed,
        "corresponding_fallback": bool(corresponding_fallback_author),
        "corresponding_fallback_author": corresponding_fallback_author,
        "venue": venue,
        "venue_display": venue_display,
        "venue_short": str(override.get("venue_short") or venue_short(venue)),
        "year": year,
        "citations": int(row.get("citations") or 0),
        "metadata": metadata,
        "description": description,
        "summary": summary,
        "detail_metadata_available": bool(metadata or description),
        "topics": topics,
        "links": links,
        "repo": repo,
        "repo_label": str(override.get("repo_label") or (previous or {}).get("repo_label") or ""),
        "github": dict((previous or {}).get("github", {})),
        "thumbnail": str(override.get("thumbnail") or (previous or {}).get("thumbnail") or ""),
        "visual_tone": (sum(ord(char) for char in display_title) % 5) + 1,
        "force_include": bool(override.get("force_include")),
        "force_exclude": bool(override.get("force_exclude")),
    }


def apply_selection(publications: list[dict[str, Any]], policy: dict[str, Any]) -> None:
    citation_order = sorted(publications, key=lambda item: (-item["citations"], -item["year"], item["title"]))
    for rank, publication in enumerate(citation_order, start=1):
        publication["citation_rank"] = rank

    min_citations = int(policy.get("lead_min_citations", 10))
    top_rank = int(policy.get("top_citation_rank", 20))
    max_authors = int(policy.get("top_cited_max_authors", 15))
    for publication in publications:
        self_author = next((author for author in publication["authors"] if author.get("is_self")), None)
        self_roles = set(self_author.get("roles", [])) if self_author else set()
        is_lead = bool(self_roles & {"first", "equal", "corresponding", "project_lead"})
        lead_and_influential = is_lead and publication["citations"] >= min_citations
        top_cited_compact_team = (
            publication["citation_rank"] <= top_rank
            and publication.get("author_count") is not None
            and publication["author_count"] <= max_authors
            and not publication.get("group_authored")
        )
        reasons = []
        if lead_and_influential:
            reasons.append("lead_author_and_influential")
        if top_cited_compact_team:
            reasons.append("top_cited_compact_team")
        selected = bool(reasons)
        if publication.pop("force_include", False):
            selected = True
            reasons.append("manual_include")
        if publication.pop("force_exclude", False):
            selected = False
            reasons = ["manual_exclude"]
        publication["selected"] = selected
        publication["selection_reasons"] = reasons


def fetch_github_project(config: dict[str, Any]) -> dict[str, Any]:
    repo = config["repo"]
    api_url = f"https://api.github.com/repos/{repo}"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    api_data: dict[str, Any] | None = None
    source = "github_api"
    try:
        api_data = json.loads(http_get(api_url, headers=headers))
    except (FetchError, json.JSONDecodeError):
        source = "github_html"

    if api_data and "stargazers_count" in api_data:
        stars = int(api_data.get("stargazers_count") or 0)
        forks = int(api_data.get("forks_count") or 0)
        return {
            "name": config.get("name") or api_data.get("name") or repo.split("/")[-1],
            "full_name": repo,
            "url": api_data.get("html_url") or f"https://github.com/{repo}",
            "description": config.get("description") or api_data.get("description") or "",
            "stars": stars,
            "stars_display": f"{stars:,}",
            "forks": forks,
            "forks_display": f"{forks:,}",
            "watchers": int(api_data.get("subscribers_count") or 0),
            "language": api_data.get("language"),
            "topics": config.get("topics", []),
            "updated_at": api_data.get("updated_at"),
            "source": source,
        }

    page_url = f"https://github.com/{repo}"
    source_html = http_get(page_url)

    def embedded_count(key: str) -> int:
        values = [int(value) for value in re.findall(rf'"{re.escape(key)}"\s*:\s*(\d+)', source_html)]
        return max(values) if values else 0

    language_match = re.search(r'"primaryLanguage"\s*:\s*\{[^{}]*"name"\s*:\s*"([^"]+)"', source_html)
    stars = embedded_count("stargazerCount")
    forks = embedded_count("forksCount")
    return {
        "name": config.get("name") or repo.split("/")[-1],
        "full_name": repo,
        "url": page_url,
        "description": config.get("description", ""),
        "stars": stars,
        "stars_display": f"{stars:,}",
        "forks": forks,
        "forks_display": f"{forks:,}",
        "watchers": embedded_count("watchersCount"),
        "language": language_match.group(1) if language_match else None,
        "topics": config.get("topics", []),
        "updated_at": None,
        "source": source,
    }


def build_data(args: argparse.Namespace) -> dict[str, Any]:
    overrides = json.loads(args.overrides.read_text(encoding="utf-8"))
    scholar_id = overrides["scholar_id"]
    self_aliases = {normalize_name(name) for name in overrides.get("self_names", [])}
    previous = load_previous(args.output)
    legacy = legacy_publications(self_aliases)
    resume_roles = load_resume_self_roles(args.resume_source)
    normalized_overrides = {
        normalize_title(title): value
        for title, value in overrides.get("publications", {}).items()
    }

    scholar_source = http_get(profile_url(scholar_id))
    scholar = parse_scholar_profile(scholar_source, scholar_id)
    page_rows = scholar["publications"]
    offset = 100
    known_ids = {row["scholar_id"] for row in page_rows}
    while len(page_rows) == 100:
        next_source = http_get(profile_url(scholar_id, offset))
        next_page = parse_scholar_profile(next_source, scholar_id)
        page_rows = [
            row for row in next_page["publications"]
            if row["scholar_id"] not in known_ids
        ]
        if not page_rows:
            break
        scholar["publications"].extend(page_rows)
        known_ids.update(row["scholar_id"] for row in page_rows)
        offset += 100
    publications: list[dict[str, Any]] = []
    total = len(scholar["publications"])
    detail_fetches = 0
    detail_failures: list[str] = []
    for index, row in enumerate(scholar["publications"], start=1):
        prior = previous.get(row.get("scholar_id", ""))
        detail: dict[str, Any] | None = None
        should_fetch_detail = not args.quick and (
            args.refresh_details
            or not prior
            or prior.get("authors_truncated")
            or not prior.get("authors")
        )
        if should_fetch_detail and (args.max_details is None or detail_fetches < args.max_details):
            try:
                detail = parse_scholar_detail(http_get(row["detail_url"]))
                detail_fetches += 1
                if args.delay:
                    time.sleep(args.delay)
            except FetchError as exc:
                print(f"warning: {exc}", file=sys.stderr)
                detail_failures.append(row.get("scholar_id", row.get("title", "unknown")))
        legacy_item = match_legacy(row["title"], legacy)
        publication_override = normalized_overrides.get(normalize_title(row["title"]), {})
        publications.append(merge_publication(
            row,
            detail,
            prior,
            legacy_item,
            publication_override,
            self_aliases,
            resume_roles.get(normalize_title(row["title"]), set()),
        ))
        if detail_fetches and detail_fetches % 10 == 0 and detail_fetches != getattr(args, "_last_reported", 0):
            print(f"Scholar details fetched: {detail_fetches}/{total}", file=sys.stderr)
            args._last_reported = detail_fetches

    apply_selection(publications, overrides.get("selection", {}))
    publications.sort(key=lambda item: (-item["year"], -item["citations"], item["title"]))

    publication_repos: dict[str, dict[str, Any]] = {}
    for publication in publications:
        repo = publication.get("repo")
        if not repo or repo in publication_repos:
            continue
        try:
            publication_repos[repo] = fetch_github_project({"repo": repo})
        except FetchError as exc:
            print(f"warning: {exc}", file=sys.stderr)
            publication_repos[repo] = dict(publication.get("github", {}))
    for publication in publications:
        repo = publication.get("repo")
        if repo:
            publication["github"] = publication_repos.get(repo, {})

    projects = []
    for project_config in overrides.get("projects", []):
        try:
            projects.append(fetch_github_project(project_config))
        except FetchError as exc:
            print(f"warning: {exc}", file=sys.stderr)
            projects.append({
                "name": project_config.get("name") or project_config["repo"].split("/")[-1],
                "full_name": project_config["repo"],
                "url": f"https://github.com/{project_config['repo']}",
                "description": project_config.get("description", ""),
                "stars": 0,
                "stars_display": "0",
                "forks": 0,
                "forks_display": "0",
                "watchers": 0,
                "language": None,
                "topics": project_config.get("topics", []),
                "updated_at": None,
                "source": "unavailable",
            })

    generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    scholar_metrics = scholar.get("metrics", {})
    full_author_count = sum(
        1 for item in publications
        if item.get("authors") and not item.get("authors_truncated")
    )
    metadata_count = sum(1 for item in publications if item.get("detail_metadata_available"))
    return {
        "generated_at": generated_at,
        "sources": {
            "scholar": profile_url(scholar_id),
            "github": "https://github.com/kennymckormick",
        },
        "profile": {
            "name": scholar.get("name") or "Haodong Duan",
            "scholar_id": scholar_id,
            "research_topics": overrides.get("research_topics", []),
            "institutions": overrides.get("institutions", []),
            "scholar_interests": scholar.get("interests", []),
            "metrics": scholar_metrics,
            "metrics_display": {
                key: (f"{value:,}" if isinstance(value, int) else "")
                for key, value in scholar_metrics.items()
            },
            "publication_count": len(publications),
            "selected_count": sum(1 for item in publications if item["selected"]),
            "full_author_count": full_author_count,
            "metadata_count": metadata_count,
        },
        "sync": {
            "profile_rows": total,
            "unique_scholar_ids": len({item["scholar_id"] for item in publications}),
            "detail_pages_fetched": detail_fetches,
            "detail_fetch_failures": detail_failures,
            "full_author_count": full_author_count,
            "metadata_count": metadata_count,
        },
        "selection_policy": overrides.get("selection", {}),
        "publications": publications,
        "projects": projects,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-source", type=Path, default=DEFAULT_RESUME_SOURCE)
    parser.add_argument("--quick", action="store_true", help="Skip Scholar detail pages")
    parser.add_argument(
        "--refresh-details", action="store_true",
        help="Refresh every Scholar detail page instead of reusing generated author data",
    )
    parser.add_argument("--delay", type=float, default=0.15, help="Delay between Scholar detail requests")
    parser.add_argument("--max-details", type=int, default=None, help="Limit detail requests for debugging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = build_data(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        output_label = str(args.output.relative_to(ROOT))
    except ValueError:
        output_label = str(args.output)
    print(
        f"Updated {output_label}: "
        f"{data['profile']['publication_count']} publications, "
        f"{data['profile']['selected_count']} selected works, "
        f"{len(data['projects'])} projects."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
