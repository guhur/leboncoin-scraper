#!/usr/bin/env python3
"""
Scrape leboncoin.fr for used servers suitable for running simulations.
Uses curl_cffi to bypass anti-bot (TLS fingerprint impersonation).
Filters: ≥20 cores, ≥100 GB RAM.

Usage:
    source .venv/bin/activate
    python scrape_servers.py
"""

import json
import re
import sys
import time
from dataclasses import dataclass
from urllib.parse import urlencode

from curl_cffi import requests as cffi_requests

# ---------------------------------------------------------------------------
# Configuration — tweak these to your needs
# ---------------------------------------------------------------------------

SEARCH_QUERIES = [
    "serveur dell poweredge",
    "serveur hp proliant",
    "serveur supermicro",
    "serveur rack xeon",
    "serveur xeon 128go",
    "serveur xeon 256go",
    "serveur epyc",
    "serveur 2u xeon",
]

MIN_CORES = 20
MIN_RAM_GB = 100
MAX_PRICE_EUR = 2000

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ServerListing:
    title: str
    price: int | None
    url: str
    list_id: str = ""
    location: str = ""
    description: str = ""
    cores: int | None = None
    ram_gb: int | None = None
    cpu_model: str = ""


# ---------------------------------------------------------------------------
# Spec extraction from text
# ---------------------------------------------------------------------------

RE_CORES = re.compile(r"(\d{1,3})\s*(?:c[oœ]urs?|cores?)\b", re.IGNORECASE)
RE_CORES_SLASH = re.compile(r"(\d{1,3})\s*c\s*/\s*(\d{1,3})\s*t", re.IGNORECASE)
RE_CPU_COUNT = re.compile(
    r"(\d)\s*(?:x|×)\s*(?:cpu|proc|xeon|epyc|e5|e7)", re.IGNORECASE
)
RE_XEON_CORES_PER = re.compile(
    r"(?:xeon|epyc|e5|e7|gold|silver|platinum)[^\n]{0,80}?(\d{1,2})\s*c(?:ores?)?",
    re.IGNORECASE,
)
RE_VCPU = re.compile(r"(\d{1,3})\s*(?:vcpu|vcore|threads?)\b", re.IGNORECASE)

RE_RAM_GB = re.compile(
    r"(\d{2,4})\s*(?:go|gb|gio)\b(?:\s*(?:de\s*)?(?:ram|ddr|mémoire|ecc))?",
    re.IGNORECASE,
)
# Only match TB when explicitly followed by "ram", "ddr", etc.
RE_RAM_TB = re.compile(
    r"(\d(?:[.,]\d)?)\s*(?:to|tb)\s*(?:de\s*)?(?:ram|ddr|mémoire|ecc)",
    re.IGNORECASE,
)
# Disk-in-TB pattern to avoid false positives (e.g. "48To" = 48TB storage, not RAM)
RE_DISK_TB = re.compile(
    r"\d{1,3}\s*(?:to|tb)\b(?!\s*(?:ram|ddr|mémoire|ecc))",
    re.IGNORECASE,
)
RE_CPU_MODEL = re.compile(
    r"((?:xeon|epyc|e5|e7|gold|silver|platinum|bronze)[\s-]*[-\w]*\d[\w-]*)",
    re.IGNORECASE,
)

# Patterns to exclude disk sizes from being confused with RAM
RE_DISK_CONTEXT = re.compile(
    r"(\d{2,4})\s*(?:go|gb|gio)\s*(?:ssd|hdd|sata|nvme|disque|stockage|disk)",
    re.IGNORECASE,
)
# Also exclude when "Go" appears right after "HDD" or "SSD" context
RE_DISK_BEFORE = re.compile(
    r"(?:ssd|hdd|sata|nvme|disque|stockage|disk)\s*\d{2,4}\s*(?:go|gb|to|tb)",
    re.IGNORECASE,
)


def extract_cores(text: str) -> int | None:
    m = RE_CORES_SLASH.search(text)
    if m:
        return int(m.group(1))

    m = RE_CORES.search(text)
    if m:
        return int(m.group(1))

    cpu_count = RE_CPU_COUNT.search(text)
    xeon_cores = RE_XEON_CORES_PER.search(text)
    if cpu_count and xeon_cores:
        return int(cpu_count.group(1)) * int(xeon_cores.group(1))

    m = RE_VCPU.search(text)
    if m:
        threads = int(m.group(1))
        if threads >= MIN_CORES:
            return threads

    return None


def extract_ram_gb(text: str) -> int | None:
    # TB of RAM (only when explicitly labeled as RAM/DDR)
    m = RE_RAM_TB.search(text)
    if m:
        val = float(m.group(1).replace(",", "."))
        return int(val * 1024)

    # Collect positions that look like disk storage, so we can skip them
    disk_positions: set[int] = set()
    for dm in RE_DISK_CONTEXT.finditer(text):
        disk_positions.add(dm.start())
    for dm in RE_DISK_BEFORE.finditer(text):
        # Mark positions near this disk mention
        for rm in RE_RAM_GB.finditer(text):
            if abs(rm.start() - dm.start()) < 30:
                disk_positions.add(rm.start())

    # Prefer matches explicitly labeled as RAM/DDR/ECC
    re_explicit_ram = re.compile(
        r"(\d{2,4})\s*(?:go|gb|gio)\s*(?:de\s*)?(?:ram|ddr\d?|ecc|mémoire)",
        re.IGNORECASE,
    )
    for m in re_explicit_ram.finditer(text):
        if m.start() in disk_positions:
            continue
        val = int(m.group(1))
        if 16 <= val <= 2048:
            return val

    # Fallback: any GB value not in disk context
    for m in RE_RAM_GB.finditer(text):
        if m.start() in disk_positions:
            continue
        val = int(m.group(1))
        if 16 <= val <= 2048:
            return val
    return None


def extract_cpu_model(text: str) -> str:
    m = RE_CPU_MODEL.search(text)
    return m.group(1).strip() if m else ""


def enrich(listing: ServerListing) -> ServerListing:
    combined = f"{listing.title} {listing.description}"
    listing.cores = extract_cores(combined)
    listing.ram_gb = extract_ram_gb(combined)
    listing.cpu_model = extract_cpu_model(combined)
    return listing


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def get_session() -> cffi_requests.Session:
    s = cffi_requests.Session(impersonate="chrome")
    return s


def search_page(session: cffi_requests.Session, query: str) -> list[dict]:
    """Fetch one search page and return raw ad dicts."""
    params = urlencode({"text": query, "price": f"50-{MAX_PRICE_EUR}"})
    url = f"https://www.leboncoin.fr/recherche?{params}"

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"    [!] Erreur réseau: {e}", file=sys.stderr)
        return []

    m = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', resp.text, re.DOTALL
    )
    if not m:
        if "restreint" in resp.text.lower():
            print("    [!] Bloqué par anti-bot", file=sys.stderr)
        return []

    try:
        data = json.loads(m.group(1))
        ads = data["props"]["pageProps"]["searchData"].get("ads", [])
        total = data["props"]["pageProps"]["searchData"].get("total", 0)
        if total > 0 and not ads:
            print(f"    ({total} total, mais pas d'annonces dans cette page)")
        return ads
    except (json.JSONDecodeError, KeyError) as e:
        print(f"    [!] Erreur parsing: {e}", file=sys.stderr)
        return []


def fetch_ad_body(session: cffi_requests.Session, list_id: str) -> str:
    """Fetch the detail page of a single ad to get the description."""
    url = f"https://www.leboncoin.fr/ad/{list_id}"
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code != 200:
            return ""
    except Exception:
        return ""

    m = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', resp.text, re.DOTALL
    )
    if not m:
        return ""

    try:
        data = json.loads(m.group(1))
        ad = data["props"]["pageProps"].get("ad", {})
        return ad.get("body", "")
    except (json.JSONDecodeError, KeyError):
        return ""


# ---------------------------------------------------------------------------
# Filtering & scoring
# ---------------------------------------------------------------------------

SERVER_KEYWORDS = [
    "serveur",
    "server",
    "poweredge",
    "proliant",
    "supermicro",
    "rack",
    "xeon",
    "epyc",
    "dell r",
    "hp dl",
    "hp ml",
    "r720",
    "r730",
    "r740",
    "r620",
    "r630",
    "r640",
    "dl360",
    "dl380",
    "dl580",
    "ml350",
]


COMPONENT_ONLY_KEYWORDS = [
    "processeur",
    "cpu seul",
    "carte mère",
    "carte mere",
    "ventirad",
    "ventilateur",
    "alimentation",
    "barrette",
    "ram seul",
    "boîtier rack",
    "boitier rack",
    "dynatron",
]


def looks_like_server(title: str) -> bool:
    t = title.lower()
    # Exclude component-only listings
    if any(kw in t for kw in COMPONENT_ONLY_KEYWORDS):
        return False
    return any(kw in t for kw in SERVER_KEYWORDS)


# Server models likely to have 20+ cores (dual-socket capable)
HIGH_END_MODELS = [
    "r720", "r730", "r740", "r630", "r640", "r620", "r910",
    "r430", "r440",
    "t620", "t630", "t640", "t430", "t440",
    "dl360 gen9", "dl360 gen10", "dl380 gen9", "dl380 gen10",
    "dl380p", "dl580",
    "c6100", "c6220",
    "rx2520", "rx300",
]


def passes_filter(listing: ServerListing) -> bool:
    if not looks_like_server(listing.title):
        return False
    if listing.cores is not None and listing.cores < MIN_CORES:
        return False
    if listing.ram_gb is not None and listing.ram_gb < MIN_RAM_GB:
        return False

    # Must have at least one useful spec parsed, OR be a known high-end model
    has_any_spec = listing.cores is not None or listing.ram_gb is not None
    title_lower = listing.title.lower()
    is_high_end = any(model in title_lower for model in HIGH_END_MODELS)

    # Also allow listings mentioning "bi-xeon", "double xeon", "2x xeon", etc.
    has_dual_cpu = bool(
        re.search(r"(?:bi|double|dual|2\s*x)\s*(?:xeon|epyc|cpu|proc)", title_lower)
    )

    return has_any_spec or is_high_end or has_dual_cpu


def score_listing(listing: ServerListing) -> float:
    score = 0.0
    if listing.cores is not None:
        score += min(listing.cores / MIN_CORES, 3.0) * 30
    if listing.ram_gb is not None:
        score += min(listing.ram_gb / MIN_RAM_GB, 3.0) * 30
    if listing.price is not None and listing.price > 0:
        if listing.price < 200:
            score += 5
        elif listing.price < 500:
            score += 15
        elif listing.price < 1000:
            score += 10
    if listing.cpu_model:
        score += 5
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"🔍 Recherche de serveurs d'occasion sur leboncoin.fr")
    print(f"   Critères: ≥{MIN_CORES} cœurs, ≥{MIN_RAM_GB} Go RAM, ≤{MAX_PRICE_EUR}€")
    print()

    session = get_session()
    seen_ids: set[str] = set()
    all_listings: list[ServerListing] = []

    for query in SEARCH_QUERIES:
        print(f"  🔎 '{query}' ...", end=" ", flush=True)
        ads = search_page(session, query)
        print(f"→ {len(ads)} résultats")

        for ad in ads:
            lid = str(ad.get("list_id", ""))
            if not lid or lid in seen_ids:
                continue
            seen_ids.add(lid)

            title = ad.get("subject", "")
            if not looks_like_server(title):
                continue

            price_data = ad.get("price", [])
            price = price_data[0] if isinstance(price_data, list) and price_data else None

            loc = ad.get("location", {})
            location = ""
            if isinstance(loc, dict):
                parts = [loc.get("city", ""), loc.get("zipcode", "")]
                location = " ".join(p for p in parts if p)

            listing = ServerListing(
                title=title,
                price=price,
                url=f"https://www.leboncoin.fr/ad/{lid}",
                list_id=lid,
                location=location,
                description=ad.get("body", ""),
            )
            listing = enrich(listing)
            all_listings.append(listing)

        time.sleep(1.5)

    # For listings without specs, fetch full descriptions
    need_detail = [l for l in all_listings if l.cores is None and l.ram_gb is None]
    if need_detail:
        print(f"\n  📄 Récupération des descriptions ({len(need_detail)} annonces)...")
        for i, listing in enumerate(need_detail):
            print(f"    [{i+1}/{len(need_detail)}] {listing.title[:65]}...", flush=True)
            listing.description = fetch_ad_body(session, listing.list_id)
            listing = enrich(listing)
            # Update in-place (dataclass is mutable)
            time.sleep(0.8)

    # Apply filter
    filtered = [l for l in all_listings if passes_filter(l)]
    filtered.sort(key=lambda l: score_listing(l), reverse=True)

    # Display
    print("\n" + "=" * 80)
    print(f"  RÉSULTATS: {len(filtered)} serveur(s) sur {len(all_listings)} annonces scannées")
    print("=" * 80)

    confirmed = []
    maybe = []
    for listing in filtered:
        if (
            listing.cores is not None
            and listing.cores >= MIN_CORES
            and listing.ram_gb is not None
            and listing.ram_gb >= MIN_RAM_GB
        ):
            confirmed.append(listing)
        else:
            maybe.append(listing)

    if confirmed:
        print(f"\n✅ CORRESPONDANCES CONFIRMÉES ({len(confirmed)}):\n")
        for i, l in enumerate(confirmed, 1):
            print(f"  {i}. {l.title}")
            print(f"     💰 {l.price}€" if l.price else "     💰 N/A")
            print(f"     🧠 {l.cpu_model or 'N/A'} — {l.cores} cœurs")
            print(f"     💾 {l.ram_gb} Go RAM")
            if l.location:
                print(f"     📍 {l.location}")
            print(f"     🔗 {l.url}")
            print()

    if maybe:
        print(f"\n⚠️  À VÉRIFIER MANUELLEMENT ({len(maybe)}):\n")
        for i, l in enumerate(maybe, 1):
            cores_str = f"{l.cores} cœurs" if l.cores else "?"
            ram_str = f"{l.ram_gb} Go" if l.ram_gb else "?"
            print(f"  {i}. {l.title}")
            print(f"     💰 {l.price}€" if l.price else "     💰 N/A")
            print(f"     🧠 {cores_str} / 💾 {ram_str}")
            if l.location:
                print(f"     📍 {l.location}")
            print(f"     🔗 {l.url}")
            print()

    if not filtered:
        print("\n  Aucun résultat correspondant. Ajustez les filtres ou les requêtes.")

    # Export JSON
    output_file = "results_servers.json"
    export = [
        {
            "title": l.title,
            "price": l.price,
            "url": l.url,
            "location": l.location,
            "cores": l.cores,
            "ram_gb": l.ram_gb,
            "cpu_model": l.cpu_model,
            "score": round(score_listing(l), 1),
            "confirmed": l in confirmed,
        }
        for l in filtered
    ]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"📁 Résultats exportés dans {output_file}")


if __name__ == "__main__":
    main()
