#!/usr/bin/env python3
"""Count commits by members of a GitHub organization.

The default scope matches GitHub's usual contribution-counting convention:
commits reachable from each public repository's default branch during the
previous six calendar months. Private repositories are always excluded. Commit
authors are attributed by GitHub login, not by the raw name or email stored in
the commit.

The script requires GitHub CLI (``gh``) to be installed and authenticated. The
token should have ``read:org`` scope if non-public organization membership
needs to be included.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import datetime as dt
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--org",
        default="hust-open-atom-club",
        help="GitHub organization (default: %(default)s)",
    )
    parser.add_argument(
        "--months",
        type=positive_int,
        default=6,
        help="number of calendar months to inspect (default: %(default)s)",
    )
    parser.add_argument(
        "--since",
        type=parse_since,
        help="override --months with an ISO 8601 date/time, e.g. 2026-02-26",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write CSV to this path instead of standard output",
    )
    parser.add_argument(
        "--exclude-forks",
        action="store_true",
        help="exclude organization-owned fork repositories",
    )
    parser.add_argument(
        "--exclude-archived",
        action="store_true",
        help="exclude archived repositories",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress and summary messages on standard error",
    )
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def parse_since(value: str) -> dt.datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an ISO 8601 date/time") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def months_ago(moment: dt.datetime, months: int) -> dt.datetime:
    month_index = moment.year * 12 + moment.month - 1 - months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def iso_utc(moment: dt.datetime) -> str:
    return moment.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def gh_api_pages(
    endpoint: str,
    fields: dict[str, str] | None = None,
    *,
    empty_on_conflict: bool = False,
) -> list[Any]:
    command = ["gh", "api", "--method", "GET", "--paginate", "--slurp", endpoint]
    for key, value in (fields or {}).items():
        command.extend(["-f", f"{key}={value}"])

    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, 4):
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            break
        except FileNotFoundError:
            raise RuntimeError("GitHub CLI (gh) is not installed or is not on PATH")
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or error.stdout.strip() or "unknown gh error"
            if empty_on_conflict and "HTTP 409" in detail:
                return []
            if attempt < 3 and retryable_api_error(detail):
                time.sleep(2 ** (attempt - 1))
                continue
            raise RuntimeError(
                f"GitHub API request failed for {endpoint} after {attempt} "
                f"attempt(s): {detail}"
            ) from error

    if result is None:  # Defensive; the loop either succeeds or raises.
        raise RuntimeError(f"GitHub API request failed for {endpoint}")

    try:
        pages = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"GitHub API returned invalid JSON for {endpoint}") from error

    return [item for page in pages for item in page]


def retryable_api_error(detail: str) -> bool:
    transient_markers = (
        "EOF",
        "HTTP 429",
        "HTTP 500",
        "HTTP 502",
        "HTTP 503",
        "HTTP 504",
        "connection reset",
        "error connecting",
        "i/o timeout",
        "TLS handshake timeout",
        "timed out",
    )
    lowered_detail = detail.casefold()
    return any(marker.casefold() in lowered_detail for marker in transient_markers)


def selected_repositories(
    repositories: Iterable[dict[str, Any]], exclude_forks: bool, exclude_archived: bool
) -> list[dict[str, Any]]:
    return [
        repository
        for repository in repositories
        if not repository.get("private", False)
        and repository.get("visibility", "public") == "public"
        and (not exclude_forks or not repository.get("fork"))
        and (not exclude_archived or not repository.get("archived"))
        and not repository.get("disabled")
    ]


def write_csv(
    destination: Path | None,
    members: Iterable[str],
    commit_counts: Counter[str],
    repository_counts: dict[str, set[str]],
) -> None:
    stream = destination.open("w", encoding="utf-8", newline="") if destination else sys.stdout
    try:
        writer = csv.writer(stream)
        writer.writerow(["member", "commits", "repositories_contributed"])
        for member in sorted(
            members, key=lambda login: (-commit_counts[login], login.casefold())
        ):
            writer.writerow(
                [member, commit_counts[member], len(repository_counts[member])]
            )
    finally:
        if destination:
            stream.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    now = dt.datetime.now(dt.timezone.utc)
    since = args.since or months_ago(now, args.months)
    since_text = iso_utc(since)

    try:
        members_data = gh_api_pages(
            f"orgs/{args.org}/members", {"role": "all", "per_page": "100"}
        )
        repositories_data = gh_api_pages(
            f"orgs/{args.org}/repos", {"type": "public", "per_page": "100"}
        )
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    canonical_members = {
        member["login"].casefold(): member["login"]
        for member in members_data
        if member.get("login")
    }
    repositories = selected_repositories(
        repositories_data, args.exclude_forks, args.exclude_archived
    )

    if not args.quiet:
        print(
            f"Counting commits since {since_text} for {len(canonical_members)} members "
            f"across {len(repositories)} public repositories...",
            file=sys.stderr,
        )

    commit_counts: Counter[str] = Counter()
    repository_counts: dict[str, set[str]] = defaultdict(set)
    non_member_commits = 0
    unattributed_commits = 0

    for index, repository in enumerate(repositories, start=1):
        full_name = repository["full_name"]
        if not args.quiet:
            print(f"[{index}/{len(repositories)}] {full_name}", file=sys.stderr)
        try:
            commits = gh_api_pages(
                f"repos/{full_name}/commits",
                {"since": since_text, "per_page": "100"},
                empty_on_conflict=True,
            )
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

        for commit in commits:
            author = commit.get("author")
            login = author.get("login") if author else None
            if not login:
                unattributed_commits += 1
                continue
            member = canonical_members.get(login.casefold())
            if not member:
                non_member_commits += 1
                continue
            commit_counts[member] += 1
            repository_counts[member].add(full_name)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, canonical_members.values(), commit_counts, repository_counts)

    if not args.quiet:
        output_name = str(args.output) if args.output else "standard output"
        print(
            f"Done: {sum(commit_counts.values())} member commits; "
            f"{non_member_commits} commits by non-members; "
            f"{unattributed_commits} commits without a linked GitHub account. "
            f"CSV written to {output_name}.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
