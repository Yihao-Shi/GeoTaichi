"""Hierarchical browsing, search, and helper-document coverage checks."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import unicodedata

from ..core.contracts import build_docs_data, build_error, build_ok
from ..core.resources import load_capability_index


STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "only",
    "or",
    "the",
    "to",
    "using",
    "with",
}

# Search vocabulary is deliberately small and mechanics-specific. It bridges
# user vocabulary to indexed API vocabulary; a returned capability still needs
# a follow-up browse/source trace before it is treated as supported.
PHRASE_ALIASES = {
    "signed distance function": "levelset sdf lsdem",
    "signed distance field": "levelset sdf lsdem",
    "signed distance fields": "levelset sdf lsdem",
    "signed-distance function": "levelset sdf lsdem",
    "signed-distance field": "levelset sdf lsdem",
    "signed-distance fields": "levelset sdf lsdem",
    "level set": "levelset sdf lsdem",
    "level-set": "levelset sdf lsdem",
    "finite element": "fem tetrahedral deformable",
    "finite-element": "fem tetrahedral deformable",
    "material points": "mpm materialpoint",
    "material point": "mpm materialpoint",
    "control points": "iga controlpoint",
    "control point": "iga controlpoint",
    "deformable particles": "soft particle deformable",
    "deformable particle": "soft particle deformable",
    "elastic particles": "soft particle deformable",
    "elastic particle": "soft particle deformable",
    "soft particles": "soft particle deformable",
    "rigid bodies": "rigid body",
    "fluid structure interaction": "fluid solid coupling contact",
    "fluid-structure interaction": "fluid solid coupling contact",
    "颗粒流体耦合": "particle fluid coupling cfdem",
    "物质点法": "mpm materialpoint",
    "物质点": "mpm materialpoint",
    "有限元": "fem tetrahedral deformable",
    "四面体": "fem tetrahedral",
    "等几何": "iga isogeometric nurbs",
    "控制点": "iga controlpoint",
    "自接触": "self contact collision",
    "离散颗粒": "dem discrete particle",
    "离散刚体": "dem discrete rigid body",
    "堆积": "packing assembly",
    "流固耦合": "fluid solid coupling contact",
    "软颗粒": "soft particle deformable",
    "柔性颗粒": "soft particle deformable",
    "刚性颗粒": "rigid particle",
    "刚体": "rigid body",
    "水平集": "levelset sdf lsdem",
    "耦合": "coupling contact",
    "碰撞": "collision contact",
    "接触": "contact collision",
    "流体": "fluid",
    "固体": "solid",
    "颗粒": "particle",
}

TOKEN_ALIASES = {
    "bodies": "body",
    "collide": "collision",
    "collides": "collision",
    "colliding": "collision",
    "collisions": "collision",
    "couple": "coupling",
    "coupled": "coupling",
    "couples": "coupling",
    "fluids": "fluid",
    "levelsets": "levelset",
    "particles": "particle",
    "solids": "solid",
}

CATEGORY_SEARCH_TERMS = {
    "mpm": "continuum solid fluid porous material point deformable free surface",
    "dem": "discrete rigid particle sphere clump levelset sdf lsdem soft particle contact collision",
    "mpdem": "dem mpm particle solid coupling contact soft rigid lagrangian",
    "cfdem": "fluid solid particle coupling contact cfd dem drag porous incompressible",
    "fem": "finite element deformable soft particle elasticity contact cloth membrane",
    "fedem": "fem dem coupling deformable soft particle rigid body levelset sdf lsdem contact collision action reaction",
    "fempm": "fem mpm coupling deformable surface particle contact ipc",
    "iga": "isogeometric nurbs solid elasticity",
    "igampm": "iga mpm coupling nurbs particle contact",
}


# These patterns identify the physical representations named in a request.
# They are intentionally solver-level rather than tied to benchmark sentences.
FEATURE_PATTERNS = {
    "mpm": (
        r"\bmpm\b",
        r"\bmaterial[\s-]*points?\b",
        r"物质点(?:法)?",
    ),
    "dem": (
        r"\b(?:ls)?dem\b",
        r"\bdiscrete\s+(?:grains?|particles?|bodies)\b",
        r"\brigid\s+(?:grains?|particles?|bodies)\b",
        r"\baffine\s+rigid\s+(?:body|bodies)\b",
        r"离散(?:刚体|颗粒)",
        r"刚性(?:水平集)?颗粒",
    ),
    "fluid": (
        r"\bcfd\b",
        r"\bfluids?\b",
        r"\bporosity\b",
        r"\bdrag\b",
        r"\bincompressible\b",
        r"流体",
        r"孔隙率",
        r"阻力",
    ),
    "fem": (
        r"\bfem\b",
        r"\bfinite[\s-]*elements?\b",
        r"\btet(?:4)?\b",
        r"\btetrahedral\b",
        r"\b(?:cloth|membrane)\b",
        r"\b(?:soft|deformable|elastic)\s+(?:particles?|bodies|solids?|surfaces?|boundar(?:y|ies))\b",
        r"有限元",
        r"四面体",
        r"(?:软|柔性)颗粒",
        r"(?:软|柔性)体",
    ),
    "iga": (
        r"\biga\b",
        r"\bisogeometric\b",
        r"\bnurbs\b",
        r"\bsplines?\b",
        r"\bcontrol\s+points?\b",
        r"等几何",
        r"控制点",
    ),
    "levelset": (
        r"\blsdem\b",
        r"\blevel[\s-]*sets?\b",
        r"\bsigned[\s-]*distance\s+(?:functions?|fields?|bodies)\b",
        r"水平集",
    ),
    "contact": (
        r"\bcontact(?:ing)?\b",
        r"\bcollid(?:e|es|ing)\b",
        r"\bcollisions?\b",
        r"\bcoupl(?:e|ed|es|ing)\b",
        r"\binteract(?:s|ing|ion)?\b",
        r"\bfeedback\b",
        r"\bexchange\s+(?:linear\s+)?momentum\b",
        r"接触",
        r"碰撞",
        r"耦合",
        r"反馈",
        r"交换动量",
        r"相互作用",
    ),
}

DIRECT_ROUTE_PATTERNS = {
    "mpdem": (r"\bmpm\s*[-/]\s*dem\b", r"\bmpdem\b"),
    "cfdem": (
        r"\bcfd\s*[-/]\s*dem\b",
        r"\bcfdem\b",
        r"\bparticle\s*[-/]\s*fluid\b",
    ),
    "fedem": (r"\bfem\s*[-/]\s*(?:ls)?dem\b", r"\bfedem\b"),
    "fempm": (r"\bfem\s*[-/]\s*mpm\b", r"\bfempm\b"),
    "igampm": (r"\biga\s*[-/]\s*mpm\b", r"\bigampm\b"),
}

FEATURE_CATEGORIES = {
    "mpm": {"mpm", "mpdem", "fempm", "igampm"},
    "dem": {"dem", "mpdem", "cfdem", "fedem"},
    "fluid": {"cfdem"},
    "fem": {"fem", "fedem", "fempm"},
    "iga": {"iga", "igampm"},
}

NEGATION_PATTERN = re.compile(
    r"(?:\b(?:no|not|without|excluding|exclude)\b|不需要|不使用|不包含|不涉及|不是|不与)"
    r"[^,.;，。；]*",
    flags=re.IGNORECASE,
)


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text).lower()
    for phrase in sorted(PHRASE_ALIASES, key=len, reverse=True):
        text = text.replace(phrase, " %s " % PHRASE_ALIASES[phrase])
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _tokens(value: str) -> set[str]:
    normalized = _normalize_text(value)
    tokens = {TOKEN_ALIASES.get(token, token) for token in normalized.split()}
    return {token for token in tokens if token not in STOP_WORDS and len(token) > 1}


def _matched_features(value: str) -> set[str]:
    value = re.sub(
        r"(?<=[^\x00-\x7f])(?=[a-z0-9])|(?<=[a-z0-9])(?=[^\x00-\x7f])",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    return {
        feature
        for feature, patterns in FEATURE_PATTERNS.items()
        if any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)
    }


def _matched_routes(value: str) -> set[str]:
    value = re.sub(
        r"(?<=[^\x00-\x7f])(?=[a-z0-9])|(?<=[a-z0-9])(?=[^\x00-\x7f])",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    return {
        route
        for route, patterns in DIRECT_ROUTE_PATTERNS.items()
        if any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)
    }


def _request_analysis(terms: str) -> Dict[str, Any]:
    """Extract positive physics, exclusions, and solver-level route evidence."""
    text = unicodedata.normalize("NFKC", terms).lower()
    negative_spans = [match.group(0) for match in NEGATION_PATTERN.finditer(text)]
    positive_text = NEGATION_PATTERN.sub(" ", text)
    negative_text = " ".join(negative_spans)

    positive_features = _matched_features(positive_text)
    direct_routes = _matched_routes(positive_text)
    negative_routes = _matched_routes(negative_text)

    # A negated compound name (for example, "not MPM-DEM") excludes that
    # route, not both physical representations from the rest of the request.
    atomic_negative_text = negative_text
    for route in negative_routes:
        for pattern in DIRECT_ROUTE_PATTERNS[route]:
            atomic_negative_text = re.sub(
                pattern, " ", atomic_negative_text, flags=re.IGNORECASE
            )
    negative_features = _matched_features(atomic_negative_text)

    excluded_categories = set(negative_routes)
    for feature in negative_features:
        excluded_categories.update(FEATURE_CATEGORIES.get(feature, set()))

    bonuses: Dict[str, float] = {}
    evidence: Dict[str, List[str]] = {}

    def add(category: str, score: float, *reasons: str) -> None:
        if category in excluded_categories:
            return
        if score > bonuses.get(category, 0.0):
            bonuses[category] = score
            evidence[category] = list(reasons)

    for route in direct_routes:
        add(route, 1400.0, "explicit_%s_route" % route)

    if {"iga", "mpm"} <= positive_features:
        add("igampm", 1200.0, "iga", "mpm", "coupling")
    if {"fem", "mpm"} <= positive_features:
        add("fempm", 1200.0, "fem", "mpm", "coupling")
    if {"fem", "dem"} <= positive_features or {
        "fem",
        "levelset",
    } <= positive_features:
        add("fedem", 1200.0, "fem", "dem", "coupling")
    if {"mpm", "dem"} <= positive_features:
        add("mpdem", 1200.0, "mpm", "dem", "coupling")
    if "fluid" in positive_features and (
        "dem" in positive_features or "contact" in positive_features
    ):
        add("cfdem", 1200.0, "fluid", "dem", "coupling")

    represented = positive_features & {"mpm", "dem", "fem", "iga", "levelset"}
    if represented <= {"mpm"} and "mpm" in represented:
        add("mpm", 800.0, "mpm")
    if represented <= {"dem", "levelset"} and represented:
        add("dem", 800.0, "dem")
    if represented <= {"fem"} and "fem" in represented:
        add("fem", 800.0, "fem")
    if represented <= {"iga"} and "iga" in represented:
        add("iga", 800.0, "iga")
    if "fluid" in positive_features and not represented:
        add("cfdem", 700.0, "fluid", "coupling")

    return {
        "positive_features": sorted(positive_features),
        "negative_features": sorted(negative_features),
        "excluded_categories": sorted(excluded_categories),
        "bonuses": bonuses,
        "evidence": evidence,
    }


def browse(index: Dict[str, Any], path: Optional[str]) -> Dict[str, Any]:
    """Browse a capability path at category, collection, or item granularity."""
    categories = index.get("categories", {})
    clean = (path or "").strip().strip("/")
    if not clean:
        entries = [
            {
                "name": name,
                "facade": data["facade"],
                "description": data["description"],
                **data["summary"],
            }
            for name, data in categories.items()
        ]
        return build_ok(build_docs_data("browse", entries, {"count": len(entries)}))

    parts = clean.split("/")
    category_name = parts[0].lower()
    if category_name not in categories:
        return build_error(
            "category_not_found",
            "Capability category %r was not found." % parts[0],
            {"available_categories": sorted(categories)},
        )
    category = categories[category_name]
    if len(parts) == 1:
        entry = {
            key: category[key] for key in ("facade", "description", "facade_source", "reference", "related", "summary")
        }
        entry["paths"] = [
            "%s/methods" % category_name,
            "%s/keys" % category_name,
            "%s/examples" % category_name,
        ]
        return build_ok(build_docs_data("browse", [entry], {"count": 1, "category": category_name}))

    section = parts[1].lower()
    section_map = {
        "methods": "public_methods",
        "keys": "configuration_keys",
        "examples": "examples",
    }
    if section not in section_map:
        return build_error(
            "section_not_found",
            "Section %r was not found in %r." % (section, category_name),
            {"available_sections": sorted(section_map)},
        )
    values = category[section_map[section]]
    if len(parts) == 2:
        entries = [{"path": value} for value in values] if section == "examples" else values
        return build_ok(
            build_docs_data(
                "browse",
                entries,
                {"count": len(entries), "category": category_name, "section": section},
            )
        )

    if section == "examples":
        return build_error("invalid_path", "Examples are a terminal collection; browse the category examples path.")
    item_name = "/".join(parts[2:])
    matches = [item for item in values if item["name"].lower() == item_name.lower()]
    if not matches:
        return build_error(
            "item_not_found",
            "%s %r was not found in %r." % (section[:-1].title(), item_name, category_name),
            {"hints": [item["name"] for item in values if item_name.lower() in item["name"].lower()][:20]},
        )
    return build_ok(
        build_docs_data(
            "browse",
            matches,
            {"count": len(matches), "category": category_name, "section": section},
        )
    )


def _search_records(index: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for category_name, category in index.get("categories", {}).items():
        records.append(
            {
                "path": category_name,
                "name": category_name,
                "kind": "category",
                "category": category_name,
                "text": "%s %s %s %s"
                % (
                    category_name,
                    category["facade"],
                    category["description"],
                    CATEGORY_SEARCH_TERMS.get(category_name, ""),
                ),
            }
        )
        for method in category["public_methods"]:
            records.append(
                {
                    "path": "%s/methods/%s" % (category_name, method["name"]),
                    "name": method["name"],
                    "kind": "method",
                    "category": category_name,
                    "text": "%s %s %s %s" % (category_name, method["name"], method["signature"], method.get("doc", "")),
                    "source": method["source"],
                    "line": method["line"],
                }
            )
        for key in category["configuration_keys"]:
            sources = " ".join(location["source"] for location in key["locations"])
            records.append(
                {
                    "path": "%s/keys/%s" % (category_name, key["name"]),
                    "name": key["name"],
                    "kind": "key",
                    "category": category_name,
                    "text": "%s %s %s" % (category_name, key["name"], sources),
                    "locations": key["locations"],
                }
            )
        for example in category["examples"]:
            records.append(
                {
                    "path": "%s/examples" % category_name,
                    "name": Path(example).name,
                    "kind": "example",
                    "category": category_name,
                    "text": "%s %s" % (category_name, example.replace("_", " ")),
                    "example": example,
                }
            )
    return records


def _score(terms: str, record: Dict[str, Any]) -> tuple[float, List[str], str]:
    normalized_query = _normalize_text(terms)
    normalized_name = _normalize_text(record["name"])
    normalized_text = _normalize_text(record["text"])
    query_tokens = _tokens(terms)
    text_tokens = _tokens(record["text"])
    matched = sorted(query_tokens & text_tokens)
    if normalized_query == normalized_name:
        return 1000.0, matched, "exact_name"
    if normalized_query in normalized_name:
        return 900.0 + 50.0 * len(normalized_query) / max(1, len(normalized_name)), matched, "name_fragment"
    if normalized_query in normalized_text:
        return 800.0 + min(99.0, len(normalized_query)), matched, "phrase"

    coverage = len(matched) / max(1, len(query_tokens))
    precision = len(matched) / max(1, len(text_tokens))
    token_score = 720.0 * coverage + 80.0 * min(1.0, precision * 4.0)
    if record["kind"] == "category" and len(query_tokens) >= 3:
        token_score += 40.0 * coverage
    fuzzy = 0.0
    if len(query_tokens) <= 2:
        fuzzy = 420.0 * SequenceMatcher(None, normalized_query, normalized_name).ratio()
    score = max(token_score, fuzzy)
    reason = "token_overlap" if token_score >= fuzzy else "fuzzy_name"
    return score, matched, reason


def query(index: Dict[str, Any], terms: str, limit: int = 10) -> Dict[str, Any]:
    """Search capabilities and return exact paths for a follow-up browse call."""
    if not terms.strip():
        return build_error("invalid_query", "Query text must not be empty.")
    if limit < 1 or limit > 100:
        return build_error("invalid_limit", "limit must be between 1 and 100")
    analysis = _request_analysis(terms)
    scored_records = []
    for record in _search_records(index):
        score, matched_terms, match_reason = _score(terms, record)
        scored_records.append((score, record, matched_terms, match_reason))

    ranked = []
    if analysis["bonuses"]:
        # Natural-language model selection ranks supported solver categories,
        # rather than allowing several methods from one category to occupy the
        # candidate list. Exact API/key queries continue to use record ranking.
        by_category: Dict[str, List[tuple[float, Dict[str, Any], List[str], str]]] = {}
        category_records = {}
        for item in scored_records:
            category = item[1]["category"]
            by_category.setdefault(category, []).append(item)
            if item[1]["kind"] == "category":
                category_records[category] = item[1]
        for category, candidates in by_category.items():
            if category in analysis["excluded_categories"]:
                continue
            best_score, _, matched_terms, match_reason = max(
                candidates,
                key=lambda item: (item[0], -len(item[1]["path"])),
            )
            bonus = analysis["bonuses"].get(category, 0.0)
            category_score = best_score + bonus
            if category_score < 250.0:
                continue
            route_evidence = analysis["evidence"].get(category, [])
            ranked.append(
                (
                    category_score,
                    category_records[category],
                    sorted(set(matched_terms) | set(route_evidence)),
                    "physics_route" if bonus else "category_rerank",
                )
            )
    else:
        ranked = [item for item in scored_records if item[0] >= 250.0]

    ranked.sort(key=lambda pair: (-pair[0], pair[1]["path"], pair[1]["name"]))
    entries = []
    for rank, (score, record, matched_terms, match_reason) in enumerate(ranked[:limit], 1):
        result = {key: value for key, value in record.items() if key != "text"}
        result.update(
            {
                "score": round(score, 2),
                "rank": rank,
                "matched_terms": matched_terms,
                "match_reason": match_reason,
            }
        )
        if analysis["bonuses"]:
            result["route_features"] = analysis["evidence"].get(
                record["category"], []
            )
        entries.append(result)
    summary: Dict[str, Any] = {
        "count": len(entries),
        "query": terms,
        "limit": limit,
        "positive_features": analysis["positive_features"],
        "negative_features": analysis["negative_features"],
        "excluded_categories": analysis["excluded_categories"],
    }
    if not entries:
        summary["hints"] = [
            "Try a facade name: mpm, dem, mpdem, cfdem, iga, or igampm.",
            "Try a shorter physical term or exact configuration-key fragment.",
            "Browse the root categories and then inspect methods or keys.",
        ]
    return build_ok(build_docs_data("query", entries, summary))


def browse_capabilities(path: Optional[str] = None, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    return browse(load_capability_index(repo_root), path)


def query_capabilities(terms: str, limit: int = 10, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    return query(load_capability_index(repo_root), terms, limit)


def audit_helper_document(
    docs_path: Path,
    categories: Optional[List[str]] = None,
    kind: str = "methods",
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Check indexed public names against the helper API reference."""
    if kind not in {"methods", "keys", "all"}:
        return build_error("invalid_kind", "kind must be methods, keys, or all")
    index = load_capability_index(repo_root)
    selected = categories or list(index.get("categories", {}))
    unknown = sorted(set(selected) - set(index.get("categories", {})))
    if unknown:
        return build_error(
            "unknown_category",
            "One or more capability categories are unknown.",
            {"unknown": unknown, "available": sorted(index.get("categories", {}))},
        )
    text = docs_path.read_text(encoding="utf-8").replace(r"\_", "_")
    results: Dict[str, Any] = {}
    missing_total = 0
    for name in selected:
        category = index["categories"][name]
        methods = category["public_methods"] if kind in {"methods", "all"} else []
        keys = category["configuration_keys"] if kind in {"keys", "all"} else []
        missing_methods = [item for item in methods if item["name"] not in text]
        missing_keys = [item for item in keys if item["name"] not in text]
        missing_total += len(missing_methods) + len(missing_keys)
        results[name] = {
            "method_count": len(methods),
            "missing_methods": missing_methods,
            "configuration_key_count": len(keys),
            "missing_configuration_keys": missing_keys,
        }
    data = {
        "source": str(docs_path),
        "kind": kind,
        "categories": results,
        "summary": {"complete": missing_total == 0, "missing_count": missing_total},
    }
    if missing_total:
        return build_error(
            "documentation_coverage_incomplete",
            "Indexed public methods or keys are missing from the helper API reference.",
            data,
        )
    return build_ok(data)
