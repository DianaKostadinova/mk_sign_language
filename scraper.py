

import re
import sys
import csv
import time
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, unquote

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
from bs4 import BeautifulSoup

# ── config ────────────────────────────────────────────────────────────────────
API_BASE    = "https://znakoven.mk/wp-json/wp/v2"
OUTPUT_DIR  = Path("videos")
BATCH_SIZE    = 100
REQUEST_DELAY = 0.1
YTDLP_FORMAT  = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
NODE_PATH     = r"C:\Program Files\nodejs\node.exe"

YOUTUBE_PATTERNS = [
    r'youtube\.com/embed/([A-Za-z0-9_-]{11})',
    r'youtube\.com/watch\?v=([A-Za-z0-9_-]{11})',
    r'youtu\.be/([A-Za-z0-9_-]{11})',
]


# ── helpers ───────────────────────────────────────────────────────────────────
def get_all_pages(session: requests.Session) -> list[dict]:
    pages, page_num = [], 1
    while True:
        resp = session.get(
            f"{API_BASE}/pages",
            params={"per_page": BATCH_SIZE, "page": page_num,
                    "_fields": "id,title,content,link"},
            timeout=30,
        )
        resp.raise_for_status()
        if not resp.content.strip():
            break
        try:
            batch = resp.json()
        except Exception:
            break
        if not batch:
            break
        pages.extend(batch)
        total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
        print(f"  Fetched page {page_num}/{total_pages} ({len(batch)} items)")
        if page_num >= total_pages:
            break
        page_num += 1
        time.sleep(REQUEST_DELAY)
    return pages


def extract_youtube_ids(html: str) -> list[str]:
    ids = []
    for pattern in YOUTUBE_PATTERNS:
        ids.extend(re.findall(pattern, html))
    seen: set[str] = set()
    return [x for x in ids if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]


def safe_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def category_from_url(url: str) -> str:
    parts = [p for p in urlparse(url).path.split("/") if p]
    if len(parts) >= 2:
        return safe_filename(unquote(parts[-2]))
    if parts:
        return safe_filename(unquote(parts[0]))
    return "misc"


def download_video(yt_url: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--format", YTDLP_FORMAT,
        "--output", str(out_path),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--merge-output-format", "mp4",
        f"--js-runtimes=node:{NODE_PATH}",
        "--remote-components", "ejs:github",
        yt_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [WARN] {result.stderr.strip()[:200]}")
        return False
    return True


# ── build sign index ──────────────────────────────────────────────────────────
def build_index(all_pages: list[dict]) -> tuple[list[dict], list[dict]]:

    # First pass: collect everything
    all_entries: list[dict] = []
    for wp_page in all_pages:
        html = wp_page.get("content", {}).get("rendered", "")
        yt_ids = extract_youtube_ids(html)
        if not yt_ids:
            continue
        sign_name = BeautifulSoup(wp_page["title"]["rendered"], "html.parser").get_text()
        cat = category_from_url(wp_page["link"])
        for yt_id in yt_ids:
            all_entries.append({
                "sign":        sign_name,
                "category":    cat,
                "youtube_id":  yt_id,
                "youtube_url": f"https://www.youtube.com/watch?v={yt_id}",
                "wp_url":      wp_page["link"],
                "label":       f"{cat}/{sign_name}",
            })

    # Group by YouTube ID
    by_id: dict[str, list[dict]] = defaultdict(list)
    for e in all_entries:
        by_id[e["youtube_id"]].append(e)

    unique_signs: list[dict] = []
    duplicate_signs: list[dict] = []

    for yt_id, entries in by_id.items():
        if len(entries) == 1:
            unique_signs.append(entries[0])
        else:
            # Multiple signs share this video — keep the first entry as
            # the canonical one, log the rest as duplicates.
            unique_signs.append(entries[0])
            for dup in entries[1:]:
                duplicate_signs.append({
                    "canonical_sign":    entries[0]["sign"],
                    "canonical_label":   entries[0]["label"],
                    "duplicate_sign":    dup["sign"],
                    "duplicate_label":   dup["label"],
                    "youtube_id":        yt_id,
                    "youtube_url":       dup["youtube_url"],
                })

    return unique_signs, duplicate_signs


# ── main ──────────────────────────────────────────────────────────────────────
def warmup_ejs_cache():
    """Download the ejs challenge solver once so parallel workers reuse it."""
    print("Warming up yt-dlp JS challenge solver ...")
    subprocess.run(
        ["yt-dlp", "--update-to", "nightly",
         f"--js-runtimes=node:{NODE_PATH}",
         "--remote-components", "ejs:github",
         "--simulate", "--quiet", "--no-warnings",
         "https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        capture_output=True,
    )


def download_one(item: tuple[int, int, dict]) -> tuple[str, str]:
    """Worker function. Returns (status, label) where status is ok/skip/fail."""
    i, total, v = item
    out_path = OUTPUT_DIR / v["category"] / (safe_filename(v["sign"]) + ".mp4")
    label = v["label"]

    if out_path.exists():
        print(f"[{i}/{total}] SKIP  {label}")
        return "skip", label

    print(f"[{i}/{total}] {label}")
    ok = download_video(v["youtube_url"], out_path)
    return ("ok" if ok else "fail"), label


def main(dry_run: bool = False, limit: int | None = None, workers: int = 4):
    session = requests.Session()
    session.headers["User-Agent"] = "znakoven-scraper/1.0"

    print("Fetching all WordPress pages ...")
    all_pages = get_all_pages(session)
    print(f"Total WP pages fetched: {len(all_pages)}")

    unique_signs, duplicate_signs = build_index(all_pages)
    print(f"Unique signs (1 video each): {len(unique_signs)}")
    print(f"Shared-video duplicates:     {len(duplicate_signs)}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    manifest = {"unique": unique_signs, "duplicates": duplicate_signs}
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dup_csv = OUTPUT_DIR / "duplicates.csv"
    with dup_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "canonical_sign", "canonical_label",
            "duplicate_sign", "duplicate_label",
            "youtube_id", "youtube_url",
        ])
        writer.writeheader()
        writer.writerows(duplicate_signs)
    print(f"Duplicates report -> {dup_csv}  ({len(duplicate_signs)} rows)")

    if dry_run:
        print("\nDry-run — no downloads. Sample unique signs:")
        for v in unique_signs[:20]:
            print(f"  [{v['category']}] {v['sign']}  ->  {v['youtube_url']}")
        if len(unique_signs) > 20:
            print(f"  ... and {len(unique_signs) - 20} more")
        return

    targets = unique_signs[:limit] if limit else unique_signs
    warmup_ejs_cache()

    work = [(i + 1, len(targets), v) for i, v in enumerate(targets)]
    ok = fail = skip = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, item): item for item in work}
        for future in as_completed(futures):
            status, _ = future.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1

    print(f"\nDone.  Downloaded: {ok}  Skipped: {skip}  Failed: {fail}")
    print(f"Videos saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape znakoven.mk sign language videos")
    parser.add_argument("--dry-run", action="store_true",
                        help="List videos without downloading")
    parser.add_argument("--limit", type=int, default=None,
                        help="Download only the first N videos (testing)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers (default: 4, max recommended: 6)")
    args = parser.parse_args()
    main(dry_run=args.dry_run, limit=args.limit, workers=args.workers)
