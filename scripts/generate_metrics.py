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
from datetime import date, datetime
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
    args = parser.parse_args()

    window_end = date.fromisoformat(args.today) if args.today else date.today()
    window_start = subtract_one_year(window_end)

    github_token = get_github_token()
    calendar_data = fetch_github_calendar(args.login, github_token, window_start, window_end)

    github_repo_shas = fetch_github_repo_shas(args.compare_repo, github_token)
    hf_counts, hf_commit_total, hf_branch_count = fetch_hf_overlay_counts(
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
        github_repo_shas=github_repo_shas,
        window_start=window_start,
        window_end=window_end,
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
) -> str:
    width = 920
    height = 330
    card_width = 136
    card_gap = 12
    card_height = 56
    card_x = 24
    card_y = 84
    cell_size = 11
    cell_gap = 3
    grid_x = 86
    grid_y = 188
    grid_step = cell_size + cell_gap

    merged_total = sum(day.merged_count for day in calendar_days)
    active_days = sum(1 for day in calendar_days if day.merged_count > 0)
    hf_active_days = sum(1 for day in calendar_days if day.hf_count > 0)
    current_streak, best_streak = compute_streaks(calendar_days)
    peak_day = max(calendar_days, key=lambda day: day.merged_count, default=None)
    max_count = peak_day.merged_count if peak_day else 0

    month_positions = build_month_positions(calendar_days, months)
    level_max = max((day.merged_count for day in calendar_days), default=0)

    stats = [
        ("Merged total", format_number(merged_total)),
        ("GitHub total", format_number(github_total)),
        ("HF added", format_number(hf_total)),
        ("HF days", format_number(hf_active_days)),
        ("Current streak", f"{current_streak} days"),
        ("Best streak", f"{best_streak} days"),
    ]

    cards = []
    for index, (label, value) in enumerate(stats):
        x = card_x + index * (card_width + card_gap)
        cards.append(
            "\n".join(
                [
                    f'<rect class="card" x="{x}" y="{card_y}" width="{card_width}" height="{card_height}" rx="12"/>',
                    f'<text class="card-label" x="{x + 14}" y="{card_y + 22}">{escape(label)}</text>',
                    f'<text class="card-value" x="{x + 14}" y="{card_y + 43}">{escape(value)}</text>',
                ]
            )
        )

    month_labels = []
    for month_name, week_index in month_positions:
        x = grid_x + week_index * grid_step
        month_labels.append(f'<text class="axis-label" x="{x}" y="{grid_y - 12}">{escape(month_name)}</text>')

    weekday_labels = []
    for label, weekday in (("Mon", 1), ("Wed", 3), ("Fri", 5)):
        y = grid_y + weekday * grid_step + 9
        weekday_labels.append(f'<text class="axis-label" x="24" y="{y}">{label}</text>')

    cells = []
    for day in calendar_days:
        x = grid_x + day.week_index * grid_step
        y = grid_y + day.weekday * grid_step
        level = intensity_level(day.merged_count, level_max)
        classes = [f"cell", f"level-{level}"]
        if day.hf_count > 0:
            classes.append("hf-overlay")
        title = build_day_title(day)
        cells.append(
            (
                f'<rect class="{" ".join(classes)}" x="{x}" y="{y}" width="{cell_size}" '
                f'height="{cell_size}" rx="2"><title>{escape(title)}</title></rect>'
            )
        )

    legend_x = width - 212
    legend_y = height - 34
    legend_cells = []
    for index in range(5):
        x = legend_x + 60 + index * 16
        legend_cells.append(
            f'<rect class="cell level-{index}" x="{x}" y="{legend_y - 10}" width="11" height="11" rx="2"/>'
        )

    peak_label = "n/a"
    if peak_day is not None:
        peak_label = f"{format_number(max_count)} on {peak_day.date.isoformat()}"

    footer_text = (
        f"Rolling window: {window_start.isoformat()} to {window_end.isoformat()}  |  "
        f"HF overlay from {hf_repo_id} across {hf_branch_count} branch"
        f"{'' if hf_branch_count == 1 else 'es'}, excluding commits already present on {compare_repo}"
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">Contribution calendar for {escape(login)}</title>
  <desc id="desc">Rolling last 12 months of GitHub contributions with Hugging Face-only commits overlaid from {escape(hf_repo_id)}.</desc>
  <style>
    :root {{
      --bg: #ffffff;
      --text: #1f2328;
      --muted: #656d76;
      --card: #f6f8fa;
      --card-stroke: #d0d7de;
      --zero: #ebedf0;
      --l1: #9be9a8;
      --l2: #40c463;
      --l3: #30a14e;
      --l4: #216e39;
      --hf: #d97706;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0d1117;
        --text: #f0f6fc;
        --muted: #8b949e;
        --card: #161b22;
        --card-stroke: #30363d;
        --zero: #161b22;
        --l1: #0e4429;
        --l2: #006d32;
        --l3: #26a641;
        --l4: #39d353;
        --hf: #f59e0b;
      }}
    }}
    svg {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }}
    .title {{
      fill: var(--text);
      font-size: 26px;
      font-weight: 700;
    }}
    .subtitle {{
      fill: var(--muted);
      font-size: 13px;
    }}
    .card {{
      fill: var(--card);
      stroke: var(--card-stroke);
    }}
    .card-label {{
      fill: var(--muted);
      font-size: 12px;
    }}
    .card-value {{
      fill: var(--text);
      font-size: 21px;
      font-weight: 700;
    }}
    .axis-label {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .cell {{
      stroke: transparent;
      stroke-width: 1.5;
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
    .hf-overlay {{
      stroke: var(--hf);
    }}
    .legend-label, .footer {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .peak {{
      fill: var(--text);
      font-size: 12px;
      font-weight: 600;
    }}
  </style>
  <text class="title" x="24" y="38">Contributions Calendar</text>
  <text class="subtitle" x="24" y="58">GitHub activity with Hugging Face-only commits layered into the same rolling 12-month graph.</text>
  {''.join(cards)}
  <text class="peak" x="24" y="164">Peak day: {escape(peak_label)}  |  Active days: {active_days}</text>
  {''.join(month_labels)}
  {''.join(weekday_labels)}
  {''.join(cells)}
  <text class="legend-label" x="{legend_x}" y="{legend_y}">Lower</text>
  {''.join(legend_cells)}
  <text class="legend-label" x="{legend_x + 146}" y="{legend_y}">Higher</text>
  <rect class="cell level-0 hf-overlay" x="{legend_x}" y="{legend_y - 10}" width="11" height="11" rx="2"/>
  <text class="legend-label" x="{legend_x + 18}" y="{legend_y}">HF overlay day</text>
  <text class="footer" x="24" y="{height - 18}">{escape(footer_text)}</text>
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
