#!/usr/bin/env python3
"""Generate a rolling 12-month contributions SVG with Hugging Face overlays."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from huggingface_hub import HfApi


GRAPHQL_ENDPOINT = "https://api.github.com/graphql"
DEFAULT_LOGIN = "topemalheiro"
DEFAULT_COMPARE_REPO = "topemalheiro/CV-Project"
DEFAULT_HF_REPO_ID = "topemalheiro/CV_Project"
DEFAULT_HF_REPO_TYPE = "model"
DEFAULT_HF_SYNC_CUTOFF = "2025-12-27"


@dataclass(frozen=True)
class CalendarDay:
    date: date
    week_index: int
    weekday: int
    github_count: int
    hf_count: int

    @property
    def merged_count(self) -> int:
        return self.github_count + self.hf_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="metrics.svg", help="Output SVG path.")
    parser.add_argument("--login", default=os.getenv("GITHUB_LOGIN", DEFAULT_LOGIN))
    parser.add_argument(
        "--compare-repo",
        default=os.getenv("GITHUB_COMPARE_REPO", DEFAULT_COMPARE_REPO),
        help="GitHub repo used to exclude already-counted commits.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=os.getenv("HF_REPO_ID", DEFAULT_HF_REPO_ID),
        help="Hugging Face repo that contributes extra commit signal.",
    )
    parser.add_argument(
        "--hf-repo-type",
        default=os.getenv("HF_REPO_TYPE", DEFAULT_HF_REPO_TYPE),
        help="Hugging Face repo type: model, dataset, or space.",
    )
    parser.add_argument(
        "--today",
        default=os.getenv("METRICS_TODAY"),
        help="Override the end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--hf-sync-cutoff",
        default=os.getenv("HF_SYNC_CUTOFF", DEFAULT_HF_SYNC_CUTOFF),
        help="Do not count HF overlay commits on or after this YYYY-MM-DD date.",
    )
    args = parser.parse_args()

    window_end = date.fromisoformat(args.today) if args.today else date.today()
    window_anchor = subtract_one_year(window_end)
    window_start = previous_sunday(window_anchor)
    hf_sync_cutoff = date.fromisoformat(args.hf_sync_cutoff) if args.hf_sync_cutoff else None

    github_token = get_github_token()
    calendar_data = fetch_github_calendar(args.login, github_token, window_start, window_end)

    github_repo_shas = fetch_github_repo_shas(args.compare_repo, github_token)
    hf_counts, hf_commit_total, hf_branch_count = fetch_hf_overlay_counts(
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
        github_repo_shas=github_repo_shas,
        window_start=window_start,
        window_end=window_end,
        hf_sync_cutoff=hf_sync_cutoff,
    )

    calendar_days = merge_calendar_days(calendar_data["weeks"], hf_counts)
    svg = render_svg(
        login=args.login,
        compare_repo=args.compare_repo,
        hf_repo_id=args.hf_repo_id,
        window_start=window_start,
        window_end=window_end,
        calendar_days=calendar_days,
        months=calendar_data["months"],
        github_total=calendar_data["totalContributions"],
        hf_total=hf_commit_total,
        hf_branch_count=hf_branch_count,
        hf_sync_cutoff=hf_sync_cutoff,
    )

    output_path = Path(args.output)
    output_path.write_text(svg, encoding="utf-8")

    merged_total = sum(day.merged_count for day in calendar_days)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "github_total": calendar_data["totalContributions"],
                "hf_total": hf_commit_total,
                "merged_total": merged_total,
                "hf_branch_count": hf_branch_count,
                "hf_sync_cutoff": hf_sync_cutoff.isoformat() if hf_sync_cutoff else None,
            },
            indent=2,
        )
    )
    return 0


def subtract_one_year(day: date) -> date:
    try:
        return day.replace(year=day.year - 1)
    except ValueError:
        return day.replace(year=day.year - 1, month=2, day=28)


def previous_sunday(day: date) -> date:
    return day - timedelta(days=(day.weekday() + 1) % 7)


def get_github_token() -> str:
    for env_name in ("GITHUB_TOKEN", "METRICS_TOKEN"):
        value = os.getenv(env_name)
        if value:
            return value

    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Unable to resolve a GitHub token. Set GITHUB_TOKEN or METRICS_TOKEN, "
            "or authenticate with `gh auth login`."
        ) from exc

    token = result.stdout.strip()
    if not token:
        raise RuntimeError("Resolved an empty GitHub token.")
    return token


def fetch_github_calendar(login: str, token: str, window_start: date, window_end: date) -> dict[str, Any]:
    query = """
query($login:String!, $from:DateTime!, $to:DateTime!) {
  user(login:$login) {
    contributionsCollection(from:$from, to:$to) {
      contributionCalendar {
        totalContributions
        weeks {
          firstDay
          contributionDays {
            contributionCount
            date
            weekday
          }
        }
        months {
          firstDay
          name
          totalWeeks
          year
        }
      }
    }
  }
}
"""
    payload = {
        "query": query,
        "variables": {
            "login": login,
            "from": f"{window_start.isoformat()}T00:00:00Z",
            "to": f"{window_end.isoformat()}T23:59:59Z",
        },
    }
    request = Request(
        GRAPHQL_ENDPOINT,
        method="POST",
        headers={
            "Authorization": f"bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "topemalheiro-metrics-generator",
        },
        data=json.dumps(payload).encode("utf-8"),
    )

    try:
        with urlopen(request) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub GraphQL request failed: {detail}") from exc
    except URLError as exc:
        raise RuntimeError("GitHub GraphQL request failed.") from exc

    if body.get("errors"):
        raise RuntimeError(f"GitHub GraphQL returned errors: {body['errors']}")

    calendar = body["data"]["user"]["contributionsCollection"]["contributionCalendar"]
    return calendar


def fetch_github_repo_shas(compare_repo: str, github_token: str) -> set[str]:
    repo_url = f"https://github.com/{compare_repo}.git"

    with tempfile.TemporaryDirectory(prefix="metrics-compare-repo-") as tmp_dir:
        bare_dir = Path(tmp_dir) / "repo.git"
        clone_attempts = [(repo_url, False)]
        if github_token:
            clone_attempts.append((build_github_auth_url(compare_repo, github_token), True))

        last_error: Exception | None = None
        for clone_url, uses_token in clone_attempts:
            if bare_dir.exists():
                shutil.rmtree(bare_dir)
            try:
                run(
                    ["git", "clone", "--quiet", "--bare", "--filter=blob:none", clone_url, str(bare_dir)],
                    redacted=uses_token,
                )
                output = run(["git", "rev-list", "--branches"], cwd=bare_dir)
                return {line.strip() for line in output.splitlines() if line.strip()}
            except Exception as exc:  # pragma: no cover - exercised through failure paths
                last_error = exc

        raise RuntimeError(
            f"Unable to fetch commit history for compare repo {compare_repo}."
        ) from last_error


def build_github_auth_url(compare_repo: str, github_token: str) -> str:
    return f"https://x-access-token:{github_token}@github.com/{compare_repo}.git"


def fetch_hf_overlay_counts(
    repo_id: str,
    repo_type: str,
    github_repo_shas: set[str],
    window_start: date,
    window_end: date,
    hf_sync_cutoff: date | None,
) -> tuple[Counter[str], int, int]:
    api = HfApi(token=os.getenv("HF_TOKEN"))
    refs = api.list_repo_refs(repo_id, repo_type=repo_type)
    branches = sorted(branch.name for branch in refs.branches)

    unique_hf_commits: dict[str, date] = {}
    for branch in branches:
        commits = api.list_repo_commits(repo_id, repo_type=repo_type, revision=branch)
        for commit in commits:
            commit_date = commit.created_at.date() if commit.created_at else None
            if commit_date is None:
                continue
            if commit.commit_id in github_repo_shas:
                continue
            if commit_date < window_start or commit_date > window_end:
                continue
            if hf_sync_cutoff is not None and commit_date >= hf_sync_cutoff:
                continue
            unique_hf_commits.setdefault(commit.commit_id, commit_date)

    per_day = Counter(commit_date.isoformat() for commit_date in unique_hf_commits.values())
    return per_day, len(unique_hf_commits), len(branches)


def merge_calendar_days(weeks: list[dict[str, Any]], hf_counts: Counter[str]) -> list[CalendarDay]:
    merged: list[CalendarDay] = []
    for week_index, week in enumerate(weeks):
        for item in week["contributionDays"]:
            day_date = date.fromisoformat(item["date"])
            merged.append(
                CalendarDay(
                    date=day_date,
                    week_index=week_index,
                    weekday=int(item["weekday"]),
                    github_count=int(item["contributionCount"]),
                    hf_count=int(hf_counts.get(item["date"], 0)),
                )
            )
    return merged


def render_svg(
    *,
    login: str,
    compare_repo: str,
    hf_repo_id: str,
    window_start: date,
    window_end: date,
    calendar_days: list[CalendarDay],
    months: list[dict[str, Any]],
    github_total: int,
    hf_total: int,
    hf_branch_count: int,
    hf_sync_cutoff: date | None,
) -> str:
    del months

    width = 480
    height = 330
    tile_width = 13.6
    tile_height = 8.0
    step_x = tile_width / 2
    step_y = tile_height / 2
    grid_origin_x = 62
    grid_origin_y = 84

    merged_total = sum(day.merged_count for day in calendar_days)
    current_streak, best_streak = compute_streaks(calendar_days)
    peak_day = max(calendar_days, key=lambda day: day.merged_count, default=None)
    max_count = peak_day.merged_count if peak_day else 0
    level_max = max((day.merged_count for day in calendar_days), default=0)
    average_per_day = merged_total / len(calendar_days) if calendar_days else 0.0
    cutoff_text = (
        f"Includes {format_number(hf_total)} HF-only commits before {hf_sync_cutoff.isoformat()}."
        if hf_total and hf_sync_cutoff is not None
        else f"Includes {format_number(hf_total)} HF-only commits from {hf_repo_id}."
        if hf_total
        else "No extra HF-only commits are currently being added."
    )

    peak_label = "n/a"
    if peak_day is not None:
        peak_label = f"{format_number(max_count)} on {peak_day.date.isoformat()}"

    cells = []
    ordered_days = sorted(calendar_days, key=lambda day: (day.week_index + day.weekday, day.week_index, day.weekday))
    for day in ordered_days:
        x = grid_origin_x + (day.week_index - day.weekday) * step_x
        y = grid_origin_y + (day.week_index + day.weekday) * step_y
        level = intensity_level(day.merged_count, level_max)
        lift = isometric_lift(day.merged_count, level_max)
        cells.append(render_isometric_tile(x, y, tile_width, tile_height, lift, level, build_day_title(day)))

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">Contribution calendar for {escape(login)}</title>
  <desc id="desc">Rolling last 12 months of GitHub contributions with pre-sync Hugging Face commits merged into the same calendar.</desc>
  <defs>
    <filter id="brightness1">
      <feComponentTransfer>
        <feFuncR type="linear" slope="0.6"/>
        <feFuncG type="linear" slope="0.6"/>
        <feFuncB type="linear" slope="0.6"/>
      </feComponentTransfer>
    </filter>
    <filter id="brightness2">
      <feComponentTransfer>
        <feFuncR type="linear" slope="0.2"/>
        <feFuncG type="linear" slope="0.2"/>
        <feFuncB type="linear" slope="0.2"/>
      </feComponentTransfer>
    </filter>
  </defs>
  <style>
    :root {{
      --bg: #ffffff;
      --text: #24292f;
      --muted: #57606a;
      --link: #0969da;
      --zero: #ebedf0;
      --l1: #9be9a8;
      --l2: #40c463;
      --l3: #30a14e;
      --l4: #216e39;
      --tile-outline: rgba(27, 31, 35, 0.08);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0d1117;
        --text: #f0f6fc;
        --muted: #8b949e;
        --link: #58a6ff;
        --zero: #161b22;
        --l1: #0e4429;
        --l2: #006d32;
        --l3: #26a641;
        --l4: #39d353;
        --tile-outline: rgba(240, 246, 252, 0.08);
      }}
    }}
    svg {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }}
    .title {{
      fill: var(--link);
      font-size: 16px;
      font-weight: 500;
    }}
    .subtitle {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .level-0 {{
      fill: var(--zero);
    }}
    .level-1 {{
      fill: var(--l1);
    }}
    .level-2 {{
      fill: var(--l2);
    }}
    .level-3 {{
      fill: var(--l3);
    }}
    .level-4 {{
      fill: var(--l4);
    }}
    .tile-top {{
      stroke: var(--tile-outline);
      stroke-width: 0.7;
    }}
    .section-title {{
      fill: var(--link);
      font-size: 14px;
      font-weight: 500;
    }}
    .stat-line {{
      fill: var(--text);
      font-size: 13px;
    }}
    .bullet {{
      fill: var(--muted);
    }}
    .footer {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .peak {{
      fill: var(--text);
      font-size: 12px;
      font-weight: 600;
    }}
  </style>
  <text class="title" x="24" y="34">Contributions calendar</text>
  <text class="subtitle" x="24" y="52">Window ends {window_end.isoformat()} and starts {window_start.isoformat()}.</text>
  <text class="section-title" x="300" y="86">Commits streaks</text>
  <circle class="bullet" cx="286" cy="107" r="3"/>
  <text class="stat-line" x="300" y="111">Current streak {current_streak} days</text>
  <circle class="bullet" cx="286" cy="126" r="3"/>
  <text class="stat-line" x="300" y="130">Best streak {best_streak} days</text>
  <text class="section-title" x="300" y="161">Commits per day</text>
  <circle class="bullet" cx="286" cy="182" r="3"/>
  <text class="stat-line" x="300" y="186">Highest in a day at {format_number(max_count)}</text>
  <circle class="bullet" cx="286" cy="201" r="3"/>
  <text class="stat-line" x="300" y="205">Average per day at ~{average_per_day:.2f}</text>
  <text class="peak" x="24" y="72">GitHub {format_number(github_total)} + HF {format_number(hf_total)} = {format_number(merged_total)}</text>
  {''.join(cells)}
  <text class="footer" x="24" y="304">{escape(cutoff_text)}</text>
  <text class="footer" x="24" y="320">HF source: branches in private models. Peak day: {escape(peak_label)}.</text>
</svg>
"""
    return svg


def build_month_positions(
    calendar_days: list[CalendarDay], months: list[dict[str, Any]]
) -> list[tuple[str, int]]:
    week_index_by_month: dict[tuple[int, int], int] = {}
    for day in calendar_days:
        key = (day.date.year, day.date.month)
        week_index_by_month.setdefault(key, day.week_index)

    positions: list[tuple[str, int]] = []
    seen_weeks: set[int] = set()
    for month in months:
        key = (int(month["year"]), month_number(month["name"]))
        week_index = week_index_by_month.get(key)
        if week_index is None or week_index in seen_weeks:
            continue
        seen_weeks.add(week_index)
        positions.append((str(month["name"]), week_index))
    return positions


def month_number(name: str) -> int:
    names = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    return names[name]


def compute_streaks(calendar_days: list[CalendarDay]) -> tuple[int, int]:
    ordered = sorted(calendar_days, key=lambda day: day.date)

    best = 0
    current = 0
    running = 0
    for day in ordered:
        if day.merged_count > 0:
            running += 1
            best = max(best, running)
        else:
            running = 0

    for day in reversed(ordered):
        if day.merged_count == 0 and current == 0:
            continue
        if day.merged_count > 0:
            current += 1
        else:
            break

    return current, best


def intensity_level(count: int, max_count: int) -> int:
    if count <= 0:
        return 0
    if max_count <= 1:
        return 1
    scaled = math.log(count + 1) / math.log(max_count + 1)
    return max(1, min(4, math.ceil(scaled * 4)))


def isometric_lift(count: int, max_count: int) -> float:
    if count <= 0 or max_count <= 0:
        return 0.0
    scaled = math.log(count + 1) / math.log(max_count + 1)
    return round(2.0 + 14.0 * scaled, 3)


def render_isometric_tile(
    x: float,
    y: float,
    width: float,
    height: float,
    lift: float,
    level: int,
    title: str,
) -> str:
    top_top = (x + width / 2, y - lift)
    top_left = (x, y + height / 2 - lift)
    top_bottom = (x + width / 2, y + height - lift)
    top_right = (x + width, y + height / 2 - lift)
    ground_left = (x, y + height / 2)
    ground_bottom = (x + width / 2, y + height)
    ground_right = (x + width, y + height / 2)

    top_path = polygon_path([top_bottom, top_left, top_top, top_right])
    parts = [f'<g class="tile level-{level}"><title>{escape(title)}</title>']

    if lift > 0:
        left_path = polygon_path([top_left, top_bottom, ground_bottom, ground_left])
        right_path = polygon_path([top_bottom, top_right, ground_right, ground_bottom])
        parts.append(f'<path class="tile-side level-{level}" filter="url(#brightness1)" d="{left_path}"/>')
        parts.append(f'<path class="tile-side level-{level}" filter="url(#brightness2)" d="{right_path}"/>')

    parts.append(f'<path class="tile-top level-{level}" d="{top_path}"/>')
    parts.append("</g>")
    return "".join(parts)


def polygon_path(points: list[tuple[float, float]]) -> str:
    formatted = " ".join(f"{format_float(x)},{format_float(y)}" for x, y in points)
    return f"M{formatted} z"


def format_float(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def build_day_title(day: CalendarDay) -> str:
    if day.hf_count:
        return (
            f"{day.date.isoformat()}: {day.github_count} GitHub + {day.hf_count} HF = "
            f"{day.merged_count} total contributions"
        )
    return f"{day.date.isoformat()}: {day.github_count} GitHub contributions"


def format_number(value: int) -> str:
    return f"{value:,}"


def escape(value: str) -> str:
    return html.escape(value, quote=True)


def run(command: list[str], cwd: Path | None = None, redacted: bool = False) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        if redacted:
            raise RuntimeError("A git command failed while using authenticated access.") from exc
        stderr = exc.stderr.strip()
        raise RuntimeError(stderr or "A subprocess command failed.") from exc
    return completed.stdout


if __name__ == "__main__":
    sys.exit(main())
