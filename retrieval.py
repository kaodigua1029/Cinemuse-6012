"""Offline movie retrieval. Public contract: run_agent(query, constraints), run_baseline(query)."""
from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


class ConfigurationError(RuntimeError):
    pass


class DataError(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


GENRES = {
    "Sci-Fi": ["sci-fi", "sci fi", "science fiction", "科幻"],
    "Action": ["action", "动作"], "Adventure": ["adventure", "冒险"],
    "Animation": ["animation", "animated", "动画"],
    "Comedy": ["comedy", "comedies", "喜剧"], "Crime": ["crime", "犯罪"],
    "Documentary": ["documentary", "documentaries", "纪录片"],
    "Drama": ["drama", "剧情"], "Family": ["family", "家庭"],
    "Fantasy": ["fantasy", "奇幻"], "History": ["history", "historical", "历史"],
    "Horror": ["horror", "恐怖"], "Music": ["music", "音乐"],
    "Mystery": ["mystery", "悬疑"], "Romance": ["romance", "romantic", "爱情"],
    "Thriller": ["thriller", "惊悚"], "War": ["war", "战争"],
    "Western": ["western", "西部"], "TV Movie": ["tv movie", "电视电影"],
}
ALIASES = {s.casefold(): g for g, aliases in GENRES.items() for s in [g, *aliases]}
NUMERIC = {"min_year": (1870, 2200), "max_year": (1870, 2200),
           "min_runtime": (1, 2000), "max_runtime": (1, 2000), "limit": (1, 20)}
ALLOWED = {*NUMERIC, "genres", "exclude_genres", "query_text", "sort", "seen_movie_ids", "exclude_sequels", "avoid_gore", "unsupported_constraints"}


def canonical_genre(value):
    if not isinstance(value, str) or value.strip().casefold() not in ALIASES:
        raise ValueError(f"Unknown genre: {value!r}")
    return ALIASES[value.strip().casefold()]


def validate_constraints(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("constraints must be an object")
    if set(value) - ALLOWED:
        raise ValueError("Unknown constraint keys: " + ", ".join(sorted(set(value) - ALLOWED)))
    result = {}
    for key, val in value.items():
        if val is None:
            continue
        if key in NUMERIC:
            low, high = NUMERIC[key]
            if type(val) is not int or not low <= val <= high:
                raise ValueError(f"{key} must be an integer from {low} to {high}")
        elif key in ("genres", "exclude_genres"):
            if not isinstance(val, list) or len(val) > len(GENRES):
                raise ValueError(f"{key} must be a list of genres")
            val = list(dict.fromkeys(canonical_genre(x) for x in val))
        elif key in ("exclude_sequels", "avoid_gore"):
            if type(val) is not bool:
                raise ValueError(f"{key} must be boolean")
        elif key == "unsupported_constraints":
            if not isinstance(val, list) or len(val)>20 or any(not isinstance(x,str) or len(x)>500 for x in val):
                raise ValueError("unsupported_constraints must be a short list of strings")
            val = list(val)
        elif key == "seen_movie_ids":
            if not isinstance(val, list) or len(val) > 10000 or any(not isinstance(x, str) or not x or len(x) > 200 for x in val):
                raise ValueError("seen_movie_ids must contain at most 10000 nonempty record IDs")
            val = list(dict.fromkeys(val))
        elif key == "sort":
            if val not in ("relevance", "newest"):
                raise ValueError("sort must be relevance or newest")
        elif not isinstance(val, str) or len(val) > 2000:
            raise ValueError("query_text must be a string of at most 2000 characters")
        result[key] = val
    for a, b in [("min_year", "max_year"), ("min_runtime", "max_runtime")]:
        if a in result and b in result and result[a] > result[b]:
            raise ValueError(f"Conflicting constraints: {a} > {b}")
    if set(result.get("genres", [])) & set(result.get("exclude_genres", [])):
        raise ValueError("A genre cannot be both required and excluded")
    return result


def validate_query(query):
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError("query must contain 1–2000 characters")


def clean_topic(text):
    """Remove request framing, retaining meaningful topic/mood words."""
    text = re.sub(r"^(?:(?:我)?(?:想要看|想看|希望看|想要|希望)|请(?:你)?|麻烦(?:你)?|帮我|给我|再推荐|推荐|再来|再|来|几部|一些|一点|一部|电影|的|[\s，,。.!！])+", "", text)
    return re.sub(r"\s+", " ", text).strip(" .,，。!！")


def parse_constraints(query, previous=None):
    """Bounded rule grammar; does not claim arbitrary natural-language understanding."""
    validate_query(query)
    c = validate_constraints(previous)
    q = query.casefold()
    notes = ["Parsed supported Chinese/English rules locally; no LLM was called."]
    if re.search(r"\b(reset|start over|clear all)\b|重新开始|清空条件", q):
        c = {}
        notes.append("Reset all previous constraints.")
    for pattern, keys in [
        (r"any genre|clear genres|不限类型|类型不限", ["genres", "exclude_genres"]),
        (r"any year|clear year|不限年份|年份不限", ["min_year", "max_year"]),
        (r"any (?:runtime|length)|no (?:runtime|length) limit|不限时长|时长不限", ["min_runtime", "max_runtime"]),
        (r"clear topic|不限主题|清空主题", ["query_text"]),
    ]:
        if re.search(pattern, q):
            for key in keys:
                c.pop(key, None)
            q = re.sub(pattern, " ", q)
    # Structured negative preferences are removed before topic extraction.
    for pattern, key, value in [
        (r"不要续集|不看续集|排除续集|\b(?:no|without|exclude) sequels?\b", "exclude_sequels", True),
        (r"续集也可以|可以是续集|允许续集|\ballow sequels?\b", "exclude_sequels", False),
        (r"不要(?:太)?血腥|不看血腥|避免血腥|\b(?:no|without|avoid) (?:gore|gory movies|graphic violence)\b", "avoid_gore", True),
        (r"血腥程度不限|血腥也可以|\ballow gore\b", "avoid_gore", False),
    ]:
        if re.search(pattern, q):
            c[key] = value
            q = re.sub(pattern, " ", q)
    if re.search(r"清空推荐记录|允许重复|\ballow repeats\b", q):
        c.pop("seen_movie_ids", None)
        q = re.sub(r"清空推荐记录|允许重复|\ballow repeats\b", " ", q)
    # Normalize hour quantities before applying the same comparison grammar as minutes.
    def hour_minutes(match):
        value = match[1]
        words = {"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,
                 "one":1,"two":2,"three":3,"four":4,"five":5,"half":0.5,"半":0.5}
        hours = words[value] if value in words else float(value)
        hours += 0.5 if match[2] or match[3] else 0
        return f"{hours * 60:g} minutes"
    q = re.sub(r"(?<![a-z0-9_.])([0-9]+(?:\.[0-9]+)?|one|two|three|four|five|half)\s*(half)?\s*(?:hours?|hrs?)\b(半)?", hour_minutes, q)
    q = re.sub(r"([0-9]+(?:\.[0-9]+)?|[一二两三四五六七八九十半])\s*个?(半)?\s*小时(半)?", hour_minutes, q)
    # Quantity is per request; other preferences and seen IDs remain conversational.
    c["limit"] = 3
    if c.get("query_text"):
        cleaned = clean_topic(c["query_text"])
        if cleaned:
            c["query_text"] = cleaned
        else:
            c.pop("query_text", None)
    positive, negative = [], []
    spans = []
    for genre, aliases in GENRES.items():
        pattern = "|".join(re.escape(a) if re.search(r"[\u4e00-\u9fff]", a)
                           else r"\b" + re.escape(a) + r"\b" for a in aliases)
        for m in re.finditer(pattern, q):
            prefix = q[max(0, m.start() - 25):m.start()]
            neg = re.search(r"(?:\bno\s+|\bnot\s+|\bwithout\s+|\bexclude\s+|不要|排除|不看)\s*$", prefix)
            (negative if neg else positive).append(genre)
            spans.append(m.span())
    if positive:
        c["genres"] = list(dict.fromkeys(positive))
        c["exclude_genres"] = [g for g in c.get("exclude_genres", []) if g not in positive]
    if negative:
        c["exclude_genres"] = list(dict.fromkeys(c.get("exclude_genres", []) + negative))
        c["genres"] = [g for g in c.get("genres", []) if g not in negative]
    # Year ranges and strict vs inclusive comparisons.
    year_rules = [
        (r"(?:after|later than)\s+(\d{4})|(?<!\d)(\d{4})\s*年?(?:之后|以后)", "min_year", 1),
        (r"(?:since|from|at least)\s+(\d{4})|(?:不早于|最早)\s*(\d{4})", "min_year", 0),
        (r"(?:before|earlier than)\s+(\d{4})|(?<!\d)(\d{4})\s*年?(?:之前|以前)", "max_year", -1),
        (r"(?:through|until|at most)\s+(\d{4})|(?:不晚于|最晚)\s*(\d{4})", "max_year", 0),
    ]
    for pattern, key, offset in year_rules:
        for m in re.finditer(pattern, q):
            c[key] = int(next(x for x in m.groups() if x is not None)) + offset
            spans.append(m.span())
    for m in re.finditer(r"(?:between\s+)?(\d{4})\s*(?:and|to|-|至|到)\s*(\d{4})", q):
        c.update(min_year=int(m[1]), max_year=int(m[2]))
        spans.append(m.span())
    for m in re.finditer(r"\bin\s+(\d{4})\b|(?<!\d)(\d{4})年(?:上映|的电影)", q):
        year = int(next(x for x in m.groups() if x is not None))
        c.update(min_year=year, max_year=year)
        spans.append(m.span())
    runtime_rules = [
        (r"(?:under|less than|shorter than|少于|不到|短于)\s*(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|分钟)", "max_runtime", -1),
        (r"(?:at most|up to|no more than|不超过|最多|≤)\s*(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|分钟)", "max_runtime", 0),
        (r"(?:over|more than|longer than|超过|长于)\s*(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|分钟)", "min_runtime", 1),
        (r"(?:at least|不少于|至少|≥)\s*(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|分钟)", "min_runtime", 0),
        (r"(\d+(?:\.\d+)?)\s*(?:minutes?|分钟)(?:以内|以下|内)", "max_runtime", 0),
    ]
    for pattern, key, offset in runtime_rules:
        for m in re.finditer(pattern, q):
            # Avoid interpreting 'no more than' as 'more than' / 不超过 as 超过.
            if key == "min_runtime" and re.search(r"(?:no\s+|不)$", q[max(0, m.start()-3):m.start()]):
                continue
            value = float(m[1])
            c[key] = (math.ceil(value) - 1 if offset == -1 else math.floor(value)) if key == "max_runtime" else (math.floor(value) + 1 if offset == 1 else math.ceil(value))
            spans.append(m.span())
    if re.search(r"more recent|newer|latest|newest|更新一点|新一点|最近|最新", q):
        c["sort"] = "newest"
        notes.append("Interpreted recency as newest-first sorting; retained existing hard limits.")
    m = re.search(r"(?:recommend|show|give me)\s+(\d+|one|two|three|four|five|six|ten)\b|([一二两三四五六七八九十]|\d+)\s*部", q)
    if m:
        val = next(x for x in m.groups() if x is not None)
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10,
                 "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        c["limit"] = int(val) if val.isdigit() else words[val]
        spans.append(m.span())
    if re.search(r"换一部|换一个|\b(?:another|different) (?:one|movie|film)\b", q):
        c["limit"] = 1
    q = re.sub(r"换一部|换一个|换几部|换一批|\b(?:another|different) (?:one|movie|film)\b", lambda m: " " * len(m[0]), q)
    # Preserve a prior topic across short follow-ups. Explicit about/关于 replaces it.
    topic = re.search(r"(?:\babout\s+|关于)(.+)", q)
    if topic:
        c["query_text"] = topic[1].strip()
    elif not c.get("query_text"):
        chars = list(q)
        for start, end in spans:
            chars[start:end] = " " * (end-start)
        remainder = "".join(chars)
        remainder = re.sub(r"\b(?:recommend|some|movies?|films?|please|me|make|them|more|recent|newer|latest|newest|and|or|the|a|of|with|preferably|no|not|without|exclude|reset|start|over|clear|all)\b|推荐|电影|最好|不要|排除|不看|重新开始|清空条件|更新一点|新一点|最近|最新|的", " ", remainder)
        remainder = clean_topic(remainder)
        if remainder:
            c["query_text"] = remainder
    notes.append("Only supported rules become hard constraints; other text is a soft topic. Inspect returned constraints.")
    return validate_constraints(c), notes


def load_movies(path=None):
    path = Path(path or os.getenv("MOVIES_CSV", ROOT / "movies.csv"))
    required = {"title", "year", "runtime", "genres", "overview"}
    movies, seen = [], set()
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not required.issubset(reader.fieldnames or []):
                raise DataError("CSV missing columns: " + ", ".join(sorted(required - set(reader.fieldnames or []))))
            for line, raw in enumerate(reader, 2):
                try:
                    if None in raw or any(raw[k] is None for k in required):
                        raise ValueError("Malformed CSV row")
                    row = {k: (v or "").strip() for k, v in raw.items()}
                    row["year"], row["runtime"] = int(row["year"]), int(row["runtime"])
                    if not 1870 <= row["year"] <= 2200 or not 1 <= row["runtime"] <= 2000:
                        raise ValueError("Year or runtime outside valid range")
                    gs = json.loads(row["genres"]) if row["genres"].startswith("[") else re.split(r"[|;/]", row["genres"])
                    if not isinstance(gs, list) or not gs:
                        raise ValueError("genres must be a non-empty list")
                    row["genres"] = list(dict.fromkeys(canonical_genre(g) for g in gs))
                    if not row["title"] or not row["overview"]:
                        raise ValueError("Empty title or overview")
                    identity = (row["title"].casefold(), row["year"])
                    if identity in seen:
                        continue
                    seen.add(identity)
                    row["record_id"] = row.get("movie_id") or f"csv:{line}"
                    row["source_url"] = row.get("source_url", "")
                    row["csv_row"] = line
                    movies.append(row)
                except (ValueError, TypeError, KeyError) as exc:
                    raise DataError(f"Invalid CSV row {line}: {exc}") from exc
    except OSError as exc:
        raise DataError("Cannot read movie CSV; check MOVIES_CSV.") from exc
    if not movies:
        raise DataError("Movie CSV contains no records")
    return movies


def matches(movie, c):
    """Multiple required genres use OR. Every numeric bound and exclusion uses AND."""
    if movie["record_id"] in c.get("seen_movie_ids", []):
        return False
    if c.get("exclude_sequels") and (movie.get("is_sequel") != "false" or not movie.get("sequel_evidence")):
        return False
    if c.get("avoid_gore"):
        return False  # No reliable content-intensity annotations in this corpus.
    return (not c.get("genres") or bool(set(c["genres"]) & set(movie["genres"]))) and not (
        set(c.get("exclude_genres", [])) & set(movie["genres"])) and all(
        movie[field] >= c[lo] if bound == "lo" else movie[field] <= c[hi]
        for field, lo, hi in [("year", "min_year", "max_year"), ("runtime", "min_runtime", "max_runtime")]
        for bound in ("lo", "hi") if (lo if bound == "lo" else hi) in c)


def tokens(text):
    # A small explicit bilingual vocabulary, not a semantic embedding model.
    for zh, en in {"太空": "space", "外星人": "alien", "时间循环": "time loop", "月球": "moon",
                   "玩具": "toys", "人工智能": "artificial intelligence", "酒店": "hotel"}.items():
        text = text.replace(zh, " " + en + " ")
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.casefold())


def rank_movies(movies, query):
    mode = os.getenv("SEARCH_MODE", "lexical")
    texts = [" ".join([m["title"], " ".join(m["genres"]), m["overview"], m.get("director", ""), m.get("cast", "")]) for m in movies]
    if mode == "semantic":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigurationError("Install requirements-semantic.txt to enable semantic retrieval") from exc
        # User supplies a local model directory, so the local backend never downloads implicitly.
        model_path = os.getenv("EMBEDDING_MODEL_PATH", "")
        if not model_path or not Path(model_path).is_dir():
            raise ConfigurationError("Set EMBEDDING_MODEL_PATH to a downloaded multilingual SentenceTransformer directory")
        model = _semantic_model(model_path)
        vectors = model.encode_document(texts, normalize_embeddings=True)
        vector = model.encode_query(query or "movie", normalize_embeddings=True)
        return [float(x) for x in vectors @ vector], "local sentence embeddings"
    if mode != "lexical":
        raise ConfigurationError("SEARCH_MODE must be lexical or semantic")
    docs = [Counter(tokens(t)) for t in texts]
    df = Counter(t for d in docs for t in d)
    idf = {t: math.log((1 + len(docs)) / (1 + n)) + 1 for t, n in df.items()}
    def vector(counter):
        return {t: (1 + math.log(n)) * idf.get(t, 1) for t, n in counter.items()}
    qv = vector(Counter(tokens(query)))
    qnorm = math.sqrt(sum(x*x for x in qv.values()))
    scores = []
    for d in docs:
        v = vector(d)
        norm = math.sqrt(sum(x*x for x in v.values()))
        scores.append(sum(x*v.get(t, 0) for t, x in qv.items()) / (norm*qnorm) if norm*qnorm else 0.0)
    return scores, "TF-IDF keyword similarity (not semantic embeddings)"


@lru_cache(maxsize=2)
def _semantic_model(path):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path, local_files_only=True)


def run_agent(query, constraints=None):
    c, trace = parse_constraints(query, constraints)
    movies = load_movies()
    trace.append(f"Loaded {len(movies)} unique CSV records.")
    trace.append(f"Corpus release-year coverage: {min(m['year'] for m in movies)}–{max(m['year'] for m in movies)}; recommendations are limited to this corpus.")
    if len(movies) < 500:
        trace.append("Dataset has fewer than 500 records; this is not the full course dataset.")
    if c.get("seen_movie_ids"):
        trace.append(f"Excluded {len(c['seen_movie_ids'])} previously recommended record IDs in this conversation.")
    if c.get("exclude_sequels"):
        trace.append("不要续集：仅保留有审核记录的非续集；续集及状态未知的影片均排除。当前标注覆盖有限。")
    if c.get("avoid_gore"):
        trace.append("无法验证‘不要太血腥’：当前数据没有可靠的血腥程度标注，因此不返回推荐。可说‘血腥程度不限’解除条件。")
    candidates = [m for m in movies if matches(m, c)]
    trace.append(f"Hard filters retained {len(candidates)} records. No constraints were relaxed.")
    if not candidates:
        trace.append("No matches; change or clear a constraint to broaden the search.")
        return {"constraints": c, "movies": [], "trace": trace}
    scores, method = rank_movies(candidates, c.get("query_text", ""))
    pairs = list(zip(candidates, scores))
    pairs.sort(key=lambda p: (-p[0]["year"], -p[1], p[0]["title"]) if c.get("sort") == "newest"
               else (-p[1], -p[0]["year"], p[0]["title"]))
    trace.append(f"Ranked candidates using {method}; sort={c.get('sort', 'relevance')}.")
    result = []
    for m, score in pairs[:c.get("limit", 3)]:
        evidence = (f"CSV row {m['csv_row']} ({m['record_id']}): {m['title']}; year={m['year']}; "
                    f"runtime={m['runtime']} min; genres={', '.join(m['genres'])}. "
                    f"Source: {m['source_url'] or 'user-provided CSV; external source unspecified'}")
        if c.get("exclude_sequels"):
            evidence += " Non-sequel curation: " + m["sequel_evidence"] + " Source: " + m.get("sequel_source_url", "")
        result.append({k: m.get(k, "") for k in ("title", "year", "runtime", "genres", "poster_url")})
        result[-1]["genres"] = " / ".join(m["genres"])
        result[-1]["poster_source_url"] = m.get("poster_source_url", "")
        result[-1]["poster_license"] = m.get("poster_license", "")
        result[-1]["poster_kind"] = m.get("poster_kind", "")
        result[-1].update(reason="Satisfies all returned hard constraints; ranked by " + method + ".",
                          evidence=evidence, source_url=m["source_url"], record_id=m["record_id"])
    c["seen_movie_ids"] = list(dict.fromkeys(c.get("seen_movie_ids", []) + [m["record_id"] for m in result]))
    trace.append(f"Verified {len(result)} recommendations against CSV metadata; reasons are deterministic templates.")
    if len(result) < c.get("limit", 3):
        trace.append("Fewer matches than requested; returned only valid matches.")
    return {"constraints": c, "movies": result, "trace": trace}


def run_baseline(query):
    """Single LLM call: no movie CSV, retrieval, agent state or tools."""
    validate_query(query)
    key, model = os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MODEL")
    if not key or not model:
        raise ConfigurationError("Baseline requires OPENAI_API_KEY and OPENAI_MODEL; no offline answer is fabricated.")
    request = Request("https://api.openai.com/v1/responses",
                      data=json.dumps({"model": model, "input": query, "store": False}).encode(),
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=60) as response:
            data = json.load(response)
        text = "\n".join(p["text"] for item in data.get("output", []) if item.get("type") == "message"
                         for p in item.get("content", []) if p.get("type") == "output_text")
        if not text:
            raise ProviderError("LLM returned no text")
        return text
    except HTTPError as exc:
        raise ProviderError(f"LLM request failed (HTTP {exc.code}); check model, credentials and quota.") from None
    except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
        raise ProviderError("LLM connection or response failed") from None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--constraints", default="{}", help="Previous returned constraints as JSON")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run_baseline(args.query) if args.baseline else run_agent(args.query, json.loads(args.constraints)), ensure_ascii=False, indent=2))
    except (ValueError, ConfigurationError, DataError, ProviderError) as exc:
        parser.exit(1, f"{exc}\n")
