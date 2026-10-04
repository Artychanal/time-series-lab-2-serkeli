
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup


APP_ID = 1091500
STORE_URL = f"https://store.steampowered.com/app/{APP_ID}/Cyberpunk_2077/"
HISTOGRAM_URL = f"https://store.steampowered.com/appreviewhistogram/{APP_ID}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)


def _request(session: requests.Session, url: str, *, params: dict[str, Any]) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Steam request failed after four attempts: {url}") from last_error


def parse_store_metadata(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    def text(selector: str) -> str | None:
        node = soup.select_one(selector)
        return node.get_text(" ", strip=True) if node else None

    title = text(".apphub_AppName") or text("title")
    if title:
        title = title.replace(" on Steam", "").replace(" у Steam", "").strip()

    developers = [node.get_text(" ", strip=True) for node in soup.select("#developers_list a")]
    review_node = soup.select_one(".game_review_summary")
    return {
        "app_id": APP_ID,
        "title": title,
        "developers": developers,
        "release_date_text": text(".release_date .date"),
        "review_summary": review_node.get_text(" ", strip=True) if review_node else None,
        "review_tooltip": review_node.get("data-tooltip-html") if review_node else None,
        "genres": [node.get_text(" ", strip=True) for node in soup.select(".details_block a[href*='/genre/']")],
        "store_url": STORE_URL,
    }


def parse_histogram(payload: dict[str, Any]) -> pd.DataFrame:
    if payload.get("success") != 1:
        raise ValueError("Steam histogram returned an unsuccessful response")
    rollups = payload.get("results", {}).get("rollups", [])
    if not rollups:
        raise ValueError("Steam histogram contains no monthly rollups")

    rows: list[dict[str, Any]] = []
    for item in rollups:
        date = datetime.fromtimestamp(int(item["date"]), tz=timezone.utc).date()
        positive = int(item["recommendations_up"])
        negative = int(item["recommendations_down"])
        total = positive + negative
        rows.append(
            {
                "date": date.isoformat(),
                "positive_reviews": positive,
                "negative_reviews": negative,
                "total_reviews": total,
                "positive_share": positive / total if total else float("nan"),
            }
        )

    frame = pd.DataFrame(rows).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    expected = pd.date_range(frame["date"].min(), frame["date"].max(), freq="MS")
    missing = expected.difference(frame["date"])
    if len(missing):
        raise ValueError(f"Missing monthly observations: {missing.strftime('%Y-%m').tolist()}")
    return frame


def collect_steam_data(data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    session.cookies.update(
        {
            "birthtime": "568022401",
            "lastagecheckage": "1-January-1988",
            "mature_content": "1",
        }
    )

    store_response = _request(session, STORE_URL, params={"l": "english", "cc": "ua"})
    metadata = parse_store_metadata(store_response.text)
    histogram_response = _request(
        session,
        HISTOGRAM_URL,
        params={"l": "all", "review_score_preference": 0, "purchase_type": "all"},
    )
    histogram_payload = histogram_response.json()
    frame = parse_histogram(histogram_payload)

    metadata.update(
        {
            "collected_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "histogram_url": histogram_response.url,
            "observations": int(len(frame)),
            "period_start": frame["date"].min().date().isoformat(),
            "period_end": frame["date"].max().date().isoformat(),
            "total_reviews_in_series": int(frame["total_reviews"].sum()),
        }
    )

    csv_path = data_dir / "cyberpunk_2077_monthly_reviews.csv"
    metadata_path = data_dir / "cyberpunk_2077_metadata.json"
    raw_path = data_dir / "steam_histogram_raw.json"
    frame.to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_path.write_text(json.dumps(histogram_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(frame)} monthly observations to {csv_path}")
    print(f"Period: {metadata['period_start']} - {metadata['period_end']}")
    print(f"Reviews represented: {metadata['total_reviews_in_series']:,}")
    return csv_path, metadata_path


if __name__ == "__main__":
    collect_steam_data(Path(__file__).resolve().parent / "data")
