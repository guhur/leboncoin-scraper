#!/usr/bin/env python3
"""
Generic leboncoin.fr scraper powered by an LLM agent.

Give it a natural language prompt describing what you're looking for,
and it will:
  1. Use Claude to extract search queries and filter criteria
  2. Scrape leboncoin.fr for matching ads
  3. Use Claude to score and rank results against your intent

Usage:
    source .venv/bin/activate
    python scrape_leboncoin.py "serveur d'occasion 20+ coeurs 100Go RAM, budget 1500€"
    python scrape_leboncoin.py "vélo électrique cargo, budget max 2000€"
    python scrape_leboncoin.py --no-llm-scoring "canapé angle cuir noir"
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode

import anthropic
from curl_cffi import requests as cffi_requests
from tap import tapify


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLAUDE_MODEL = "claude-sonnet-4-20250514"
BATCH_SIZE = 10
REQUEST_TIMEOUT = 45
DELAY_BETWEEN_SEARCHES = 2.0
DELAY_BETWEEN_FETCHES = 1.0
MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Listing:
    title: str
    price: int | None
    url: str
    list_id: str = ""
    location: str = ""
    description: str = ""
    llm_score: int = 0
    llm_reason: str = ""
    extracted_specs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def call_claude(system: str, prompt: str, *, max_tokens: int = 1024) -> str:
    resp = get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def parse_json_response(raw: str) -> dict | list:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Search plan (LLM extracts queries + filters from user prompt)
# ---------------------------------------------------------------------------

PLAN_SYSTEM = """\
You are a search assistant for leboncoin.fr (French classifieds).
Extract structured search parameters from the user's request.
Always respond with valid JSON only, no markdown fences."""

PLAN_PROMPT = """\
The user wants to find something on leboncoin.fr. Analyze their request and return JSON.

CRITICAL rules for "queries":
- Leboncoin search = keyword matching. Keep queries SHORT: 2-4 words max.
- Use specific product names, brands, models. E.g. "dell poweredge", "vélo cargo électrique".
- Generate 6-10 varied queries mixing brands, synonyms, model names.
- Do NOT put numeric specs in queries — use numeric_filters instead.

CRITICAL rules for "exclude_keywords":
- Exclude common false positives for the item type.
- For "serveur": exclude job listings ("H/F", "serveuse", "restaurant", "emploi", "CDI", "CDD").
- Always exclude broken items: "HS", "pour pièces", "ne fonctionne pas", "en panne".

Return:
{{
  "queries": ["2-4 word queries, 6-10 of them"],
  "max_price": <int or null>,
  "min_price": <int or null>,
  "must_have_keywords": ["1-3 broad terms the item MUST mention"],
  "nice_to_have_keywords": ["bonus relevance terms"],
  "exclude_keywords": ["terms indicating irrelevant ads"],
  "numeric_filters": [
    {{
      "name": "<human name, e.g. RAM>",
      "unit": "<e.g. Go>",
      "min": <number or null>,
      "max": <number or null>,
      "extraction_patterns": ["Python regex with (?P<value>\\\\d+), e.g. '(?P<value>\\\\d+)\\\\s*Go'"]
    }}
  ],
  "description": "<one-line summary>"
}}

User request: {prompt}"""


def build_search_plan(user_prompt: str) -> dict:
    raw = call_claude(PLAN_SYSTEM, PLAN_PROMPT.format(prompt=user_prompt))
    return parse_json_response(raw)


# ---------------------------------------------------------------------------
# Batch scoring (LLM rates ads against user intent)
# ---------------------------------------------------------------------------

SCORE_SYSTEM = """\
You are a relevance scorer for classified ads on leboncoin.fr.
Given a user's intent and a batch of ads, rate each 0-100.
Be strict: 70+ = clearly meets requirements, 40-69 = partial match, <40 = poor match, 0 = irrelevant.
Always respond with a JSON array only."""

SCORE_PROMPT = """\
User wants: {description}

Filters:
{filters_summary}

Ads:
{ads_block}

Return a JSON array (same order), each element:
{{"id": <ad number>, "score": <0-100>, "reason": "<one-line>", "extracted_specs": {{<key-value specs>}}}}"""


def build_filters_summary(plan: dict) -> str:
    lines = []
    if plan.get("max_price"):
        lines.append(f"- Max price: {plan['max_price']}€")
    if plan.get("min_price"):
        lines.append(f"- Min price: {plan['min_price']}€")
    for nf in plan.get("numeric_filters", []):
        parts = [f"- {nf['name']}:"]
        if nf.get("min") is not None:
            parts.append(f"≥{nf['min']} {nf.get('unit', '')}")
        if nf.get("max") is not None:
            parts.append(f"≤{nf['max']} {nf.get('unit', '')}")
        lines.append(" ".join(parts))
    return "\n".join(lines) or "None"


def score_ads_batch(ads: list[Listing], plan: dict) -> list[dict]:
    ads_lines = []
    for i, ad in enumerate(ads):
        body = (ad.description or "")[:300].replace("\n", " ")
        price = f"{ad.price}€" if ad.price else "N/A"
        ads_lines.append(
            f"[Ad {i+1}] Title: {ad.title} | Price: {price} "
            f"| Location: {ad.location or 'N/A'} | Desc: {body}"
        )

    prompt = SCORE_PROMPT.format(
        description=plan.get("description", ""),
        filters_summary=build_filters_summary(plan),
        ads_block="\n".join(ads_lines),
    )

    try:
        raw = call_claude(SCORE_SYSTEM, prompt, max_tokens=2048)
        return parse_json_response(raw)
    except Exception as e:
        print(f"    [!] Scoring error: {e}", file=sys.stderr)
        return [{"id": i + 1, "score": 0, "reason": "error"} for i in range(len(ads))]


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------


def get_session() -> cffi_requests.Session:
    return cffi_requests.Session(impersonate="chrome")


def fetch_with_retry(
    session: cffi_requests.Session, url: str, *, retries: int = MAX_RETRIES
) -> cffi_requests.Response | None:
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            # Don't retry 4xx errors — they won't resolve
            if 400 <= resp.status_code < 500:
                return None
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt < retries:
                wait = 3 * (attempt + 1)
                print(f" [retry in {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                print(f" [!] {e}", file=sys.stderr)
    return None


def extract_ads_from_html(html: str) -> list[dict]:
    m = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
        return data["props"]["pageProps"]["searchData"].get("ads", [])
    except (json.JSONDecodeError, KeyError):
        return []


def search_leboncoin(
    session: cffi_requests.Session,
    query: str,
    min_price: int = 0,
    max_price: int | None = None,
) -> list[dict]:
    params: dict[str, str] = {"text": query}
    if min_price or max_price:
        params["price"] = f"{min_price or 0}-{max_price or ''}"

    url = f"https://www.leboncoin.fr/recherche?{urlencode(params)}"
    resp = fetch_with_retry(session, url)
    if not resp:
        return []

    if "restreint" in resp.text.lower():
        print(" [!] Blocked", file=sys.stderr)
        return []

    return extract_ads_from_html(resp.text)


def fetch_ad_body(session: cffi_requests.Session, url: str) -> str:
    resp = fetch_with_retry(session, url)
    if not resp:
        return ""
    m = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', resp.text, re.DOTALL
    )
    if not m:
        return ""
    try:
        data = json.loads(m.group(1))
        return data["props"]["pageProps"].get("ad", {}).get("body", "")
    except (json.JSONDecodeError, KeyError):
        return ""


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def prefilter(title: str, body: str, plan: dict) -> bool:
    title_lower = title.lower()
    combined = f"{title} {body}".lower()

    for kw in plan.get("exclude_keywords", []):
        if kw.lower() in title_lower:
            return False

    must = plan.get("must_have_keywords", [])
    if must and not any(kw.lower() in combined for kw in must):
        return False

    return True


def extract_numeric(text: str, nf: dict) -> float | None:
    for pattern in nf.get("extraction_patterns", []):
        try:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                val = m.group("value").replace(",", ".").replace(" ", "")
                return float(val)
        except (re.error, IndexError, ValueError):
            continue
    return None


def passes_numeric_filters(text: str, plan: dict) -> bool:
    """Return False if any numeric filter is violated. True otherwise."""
    for nf in plan.get("numeric_filters", []):
        val = extract_numeric(text, nf)
        if val is None:
            continue
        if nf.get("min") is not None and val < nf["min"]:
            return False
        if nf.get("max") is not None and val > nf["max"]:
            return False
    return True


# ---------------------------------------------------------------------------
# Parse raw ad dict into Listing
# ---------------------------------------------------------------------------


def parse_ad(ad: dict) -> Listing:
    lid = str(ad.get("list_id", ""))
    price_data = ad.get("price", [])
    price = price_data[0] if isinstance(price_data, list) and price_data else None

    loc = ad.get("location", {})
    location = ""
    if isinstance(loc, dict):
        parts = [loc.get("city", ""), loc.get("zipcode", "")]
        location = " ".join(p for p in parts if p)

    # Use the URL from the ad data (includes category slug)
    url = ad.get("url", "")
    if not url:
        url = f"https://www.leboncoin.fr/ad/{lid}"

    return Listing(
        title=ad.get("subject", ""),
        price=price,
        url=url,
        list_id=lid,
        location=location,
        description=ad.get("body", ""),
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_plan(plan: dict) -> None:
    print(f"\n📋 Plan de recherche:")
    print(f"   {plan.get('description', '')}")
    print(f"   Requêtes: {', '.join(plan.get('queries', []))}")
    if plan.get("max_price"):
        print(f"   Prix max: {plan['max_price']}€")
    for nf in plan.get("numeric_filters", []):
        parts = [f"   {nf['name']}:"]
        if nf.get("min") is not None:
            parts.append(f"≥{nf['min']}")
        if nf.get("max") is not None:
            parts.append(f"≤{nf['max']}")
        parts.append(nf.get("unit", ""))
        print(" ".join(parts))
    if plan.get("exclude_keywords"):
        print(f"   Exclusions: {', '.join(plan['exclude_keywords'])}")
    print()


def print_listing(l: Listing, idx: int) -> None:
    price = f"{l.price}€" if l.price else "N/A"
    score = f" [{l.llm_score}/100]" if l.llm_score else ""
    print(f"  {idx}. {l.title}")
    print(f"     💰 {price}{score}")
    if l.llm_reason:
        print(f"     💬 {l.llm_reason}")
    if l.extracted_specs:
        specs = ", ".join(f"{k}: {v}" for k, v in l.extracted_specs.items())
        print(f"     📐 {specs}")
    if l.location:
        print(f"     📍 {l.location}")
    print(f"     🔗 {l.url}")
    print()


def print_listing_compact(l: Listing, idx: int) -> None:
    price = f"{l.price}€" if l.price else "N/A"
    score = f" [{l.llm_score}/100]" if l.llm_score else ""
    print(f"  {idx}. {l.title[:70]} — {price}{score}")
    print(f"     {l.url}")


def print_results(filtered: list[Listing], *, scored: bool) -> None:
    print("\n" + "=" * 80)
    print(f"  RÉSULTATS: {len(filtered)} annonce(s)")
    print("=" * 80)

    if not filtered:
        print("\n  Aucun résultat. Essayez une requête différente.")
        return

    if not scored:
        print(f"\n🏆 RÉSULTATS ({len(filtered)}):\n")
        for i, l in enumerate(filtered, 1):
            print_listing(l, i)
        return

    great = [l for l in filtered if l.llm_score >= 70]
    decent = [l for l in filtered if 40 <= l.llm_score < 70]
    rest = [l for l in filtered if l.llm_score < 40]

    if great:
        print(f"\n🏆 MEILLEURES CORRESPONDANCES ({len(great)}):\n")
        for i, l in enumerate(great, 1):
            print_listing(l, i)

    if decent:
        print(f"\n👍 POTENTIELLEMENT INTÉRESSANT ({len(decent)}):\n")
        for i, l in enumerate(decent, 1):
            print_listing(l, i)

    if rest:
        print(f"\n📋 AUTRES ({len(rest)}):\n")
        for i, l in enumerate(rest, 1):
            print_listing_compact(l, i)


def export_results(filtered: list[Listing], output: str) -> None:
    data = [
        {
            "title": l.title,
            "price": l.price,
            "url": l.url,
            "location": l.location,
            "score": l.llm_score,
            "reason": l.llm_reason,
            "specs": l.extracted_specs,
        }
        for l in filtered
    ]
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n📁 Résultats exportés dans {output}")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def scrape_all(
    session: cffi_requests.Session, plan: dict
) -> list[Listing]:
    """Run all search queries and collect unique listings."""
    seen_ids: set[str] = set()
    listings: list[Listing] = []
    min_price = plan.get("min_price") or 0
    max_price = plan.get("max_price")

    for query in plan.get("queries", []):
        print(f"  🔎 '{query}' ...", end="", flush=True)
        ads = search_leboncoin(session, query, min_price=min_price, max_price=max_price)
        print(f" → {len(ads)} résultats")

        for ad in ads:
            listing = parse_ad(ad)
            if not listing.list_id or listing.list_id in seen_ids:
                continue
            seen_ids.add(listing.list_id)

            if not prefilter(listing.title, listing.description, plan):
                continue

            if not passes_numeric_filters(listing.title, plan):
                continue

            listings.append(listing)

        time.sleep(DELAY_BETWEEN_SEARCHES)

    return listings


def fetch_descriptions(
    session: cffi_requests.Session, listings: list[Listing], max_fetch: int
) -> None:
    """Fetch full descriptions for listings that don't have one."""
    need = [l for l in listings if not l.description][:max_fetch]
    if not need:
        return

    print(f"  📄 Récupération des descriptions ({len(need)} annonces)...")
    for i, listing in enumerate(need):
        print(f"    [{i+1}/{len(need)}] {listing.title[:65]}...", flush=True)
        listing.description = fetch_ad_body(session, listing.url)
        time.sleep(DELAY_BETWEEN_FETCHES)


def apply_filters(listings: list[Listing], plan: dict) -> list[Listing]:
    """Re-apply prefilter + numeric filters with full descriptions."""
    return [
        l
        for l in listings
        if prefilter(l.title, l.description, plan)
        and passes_numeric_filters(f"{l.title} {l.description}", plan)
    ]


def score_with_llm(listings: list[Listing], plan: dict, max_score: int) -> None:
    """Score listings using Claude in batches."""
    to_score = listings[:max_score]
    total_batches = (len(to_score) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n🤖 Évaluation par IA de {len(to_score)} annonces ({total_batches} lots)...")

    for batch_start in range(0, len(to_score), BATCH_SIZE):
        batch = to_score[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        print(f"    Lot {batch_num}/{total_batches}...", flush=True)

        results = score_ads_batch(batch, plan)
        for j, listing in enumerate(batch):
            if j < len(results):
                r = results[j]
                listing.llm_score = r.get("score", 0)
                listing.llm_reason = r.get("reason", "")
                listing.extracted_specs = r.get("extracted_specs", {})
            print(f"      {listing.title[:55]}... [{listing.llm_score}/100]")

        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    prompt: str = "",
    max_results: int = 30,
    no_llm_scoring: bool = False,
    output: str = "results.json",
) -> None:
    """Search leboncoin.fr with a natural language prompt.

    Args:
        prompt: What are you looking for? (natural language, in French or English)
        max_results: Max results to score with LLM
        no_llm_scoring: Skip per-ad LLM scoring (faster, less precise)
        output: Output JSON file path
    """
    if not prompt:
        prompt = input("🔍 Que recherchez-vous sur leboncoin ? > ")

    # 1. Extract search plan
    print(f"\n🤖 Analyse de votre demande...")
    plan = build_search_plan(prompt)
    print_plan(plan)

    # 2. Scrape
    session = get_session()
    listings = scrape_all(session, plan)
    print(f"\n  📊 {len(listings)} annonces après pré-filtrage")

    # 3. Fetch missing descriptions
    fetch_descriptions(session, listings, max_fetch=max_results * 2)

    # 4. Re-filter with full text
    filtered = apply_filters(listings, plan)
    print(f"  📊 {len(filtered)} annonces après filtrage complet")

    # 4b. Pre-sort: prioritize ads that mention specs / nice-to-have keywords
    #     so the LLM scores the most promising ones first
    def presort_key(l: Listing) -> int:
        combined = f"{l.title} {l.description}".lower()
        score = 0
        for kw in plan.get("nice_to_have_keywords", []):
            if kw.lower() in combined:
                score += 10
        # Boost ads mentioning numbers that look like specs (e.g. "128Go", "256Go")
        for nf in plan.get("numeric_filters", []):
            val = extract_numeric(combined, nf)
            if val is not None:
                score += 20
        return -score  # negative for ascending sort

    filtered.sort(key=presort_key)

    # 5. LLM scoring
    if not no_llm_scoring and filtered:
        score_with_llm(filtered, plan, max_results)
        filtered.sort(key=lambda l: l.llm_score, reverse=True)
    else:
        filtered.sort(key=lambda l: l.price or 99999)

    # 6. Display & export
    print_results(filtered, scored=not no_llm_scoring)
    export_results(filtered, output)


if __name__ == "__main__":
    tapify(main)
