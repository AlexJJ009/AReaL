#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manage the two development backlog Markdown files (Python 3.12, Unix).

Copy of phai-relay tools/backlog.py (2026-10-24); only the file headers differ.
Call it through scripts/research/backlog, which passes --dir.

Usage: python backlog.py --dir DIRECTORY COMMAND [OPTIONS]
  next
  add --title TITLE --found SOURCE [--status STATUS] [--body-file FILE]
  set-status N STATUS
  edit N [--title TITLE] [--body-file FILE|-]
  close N --resolution TEXT
  list [--all]
  check
DIRECTORY holds BACKLOG.md and BACKLOG-done.md.
Edit replaces an open item's title and/or body; FILE=- reads the body from stdin.
Closed items cannot be edited.

All cooperating readers/writers lock .backlog.lock. Each replacement is atomic.
A close replaces the done file first: interruption between replacements leaves
a duplicate that check reports, preserving the item for manual recovery.
"""

import argparse
import fcntl
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

OPEN_HEADER = "# AReaL research backlog\n\nOpen work. Manage items with scripts/research/backlog.\n\n"
DONE_HEADER = (
    "# AReaL research backlog — done\n\nCompleted work, archived in closure order.\n\n"
)
NAMES = ("BACKLOG.md", "BACKLOG-done.md")
HEADERS = (OPEN_HEADER, DONE_HEADER)
STATUSES = ("OPEN", "IN PROGRESS", "PARTIAL")
HEADING = re.compile(r"## ([1-9][0-9]*)\. (\S[^\n]*)\n")


@dataclass
class Item:
    number: int
    title: str
    text: str
    status: str


def item_starts(text: str) -> list[int]:
    """Find level-two headings outside Markdown backtick and tilde fences."""
    starts = []
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence is not None:
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not marker[2].strip()
            ):
                fence = None
        elif marker and (marker[1][0] != "`" or "`" not in marker[2]):
            fence = marker[1]
        elif re.match(r"^##(?:\s|$)", line):
            starts.append(offset)
        offset += len(line)
    return starts


def parse(text: str, name: str, closed: bool) -> list[Item]:
    """Validate a document without rewriting its free Markdown bodies."""
    header = HEADERS[int(closed)]
    if not text.startswith(header):
        raise ValueError(f"{name}: invalid fixed header")
    items = []
    starts = item_starts(text)
    preamble = text[len(header) : starts[0] if starts else len(text)]
    if preamble.strip():
        raise ValueError(f"{name}: unexpected content before first item")
    for index, start in enumerate(starts):
        block = text[
            start : starts[index + 1] if index + 1 < len(starts) else len(text)
        ]
        match = HEADING.match(block)
        if not match:
            raise ValueError(f"{name}: invalid heading: {block.splitlines()[0]}")
        number, title = match.groups()
        metadata = block[match.end() :].split("\n\n", 1)[0].splitlines()
        statuses = [
            line.removeprefix("Status: ")
            for line in metadata
            if line.startswith("Status: ")
        ]
        if not closed and (len(statuses) != 1 or statuses[0] not in STATUSES):
            raise ValueError(
                f"{name}: item {number} needs one Status: {' | '.join(STATUSES)}"
            )
        if closed and not re.fullmatch(
            r"Closed: \d{4}-\d{2}-\d{2} - \S.*", metadata[0] if metadata else ""
        ):
            raise ValueError(
                f"{name}: item {number} needs Closed: YYYY-MM-DD - resolution after heading"
            )
        found = [line for line in metadata if line.startswith("Found: ")]
        if len(found) != 1 or not re.fullmatch(
            r"Found: \d{4}-\d{2}-\d{2}, \S.*", found[0]
        ):
            raise ValueError(f"{name}: item {number} needs Found: YYYY-MM-DD, source")
        for line in [found[0], *([metadata[0]] if closed else [])]:
            try:
                date.fromisoformat(line.split(": ", 1)[1][:10])
            except ValueError as error:
                raise ValueError(
                    f"{name}: item {number} has invalid date: {line}"
                ) from error
        items.append(
            Item(int(number), title, block, "CLOSED" if closed else statuses[0])
        )
    return items


def validate(documents: list[str]) -> list[list[Item]]:
    groups = [
        parse(text, name, bool(i))
        for i, (text, name) in enumerate(zip(documents, NAMES, strict=True))
    ]
    seen = set()
    for group in groups:
        for item in group:
            if item.number in seen:
                raise ValueError(f"duplicate item number: {item.number}")
            seen.add(item.number)
    return groups


def replace_documents(directory: Path, before: list[str], after: list[str]) -> None:
    """Check the whole candidate before staging any replacements under the lock."""
    validate(after)
    staged = []
    try:
        for name, old, new in zip(NAMES, before, after, strict=True):
            if old == new and (directory / name).exists():
                continue
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=directory,
                prefix=f".{name}.",
                delete=False,
            ) as stream:
                staged.append((Path(stream.name), directory / name))
                stream.write(new)
                stream.flush()
                os.fsync(stream.fileno())
        # Archive first so an interrupted close retains a detectable duplicate.
        for temporary, target in reversed(staged):
            os.replace(temporary, target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def single_line(value: str) -> str:
    if not value.strip() or "\n" in value or "\r" in value:
        raise argparse.ArgumentTypeError("must be a non-empty single line")
    return value.strip()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    result.add_argument(
        "--dir", type=Path, required=True, help="directory containing backlog files"
    )
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("next", help="print max number across both files + 1")
    add = commands.add_parser("add", help="append an open item and print its number")
    add.add_argument("--title", type=single_line, required=True)
    add.add_argument("--found", type=single_line, required=True, metavar="SOURCE")
    add.add_argument("--status", choices=STATUSES, default="OPEN")
    add.add_argument("--body-file", type=Path)
    status = commands.add_parser("set-status", help="change an open item's status")
    status.add_argument("number", type=int, metavar="N")
    status.add_argument("status", choices=STATUSES, metavar="S")
    edit = commands.add_parser("edit", help="replace an open item's title and/or body")
    edit.add_argument("number", type=int, metavar="N")
    edit.add_argument("--title", type=single_line)
    edit.add_argument(
        "--body-file", type=Path, help="replacement body file, or - for stdin"
    )
    close = commands.add_parser("close", help="move an item to the end of the archive")
    close.add_argument("number", type=int, metavar="N")
    close.add_argument("--resolution", type=single_line, required=True)
    listing = commands.add_parser("list", help="print the open item table")
    listing.add_argument("--all", action="store_true", help="include closed items")
    commands.add_parser("check", help="validate headings, numbering and metadata")
    return result


def append_item(document: str, item: str) -> str:
    """Append with a blank separator, preserving all existing Markdown bytes."""
    separator = (
        "" if document.endswith("\n\n") else "\n" if document.endswith("\n") else "\n\n"
    )
    return document + separator + item + ("" if item.endswith("\n\n") else "\n\n")


def update(args, documents: list[str], groups: list[list[Item]]) -> int | None:
    if args.command == "add":
        number = max((item.number for group in groups for item in group), default=0) + 1
        body = args.body_file.read_text(encoding="utf-8") if args.body_file else ""
        # A body must not inject another item or override reserved metadata.
        if item_starts(body):
            raise ValueError("body must not contain level-two item headings")
        item = (
            f"## {number}. {args.title}\nStatus: {args.status}\n"
            f"Found: {date.today().isoformat()}, {args.found}\n\n{body}"
        )
        documents[0] = append_item(documents[0], item)
        return number
    item = next((item for item in groups[0] if item.number == args.number), None)
    if item is None:
        raise ValueError(f"item {args.number} is missing or already closed")
    if args.command == "edit":
        if args.title is None and args.body_file is None:
            raise ValueError("edit requires --title and/or --body-file")
        changed = item.text
        if args.body_file is not None:
            body = (
                sys.stdin.read()
                if args.body_file == Path("-")
                else args.body_file.read_text(encoding="utf-8")
            )
            if item_starts(body):
                raise ValueError("body must not contain level-two item headings")
            # An unclosed fence would hide the next item's heading on reparse.
            if not item_starts(body + "\n## 1. sentinel\n"):
                raise ValueError("body must not contain an unclosed code fence")
            headers = item.text.split("\n\n", 1)[0]
            changed = headers + "\n\n" + body
            if not changed.endswith("\n\n"):
                changed += "\n" if changed.endswith("\n") else "\n\n"
        if args.title is not None:
            _, rest = changed.split("\n", 1)
            changed = f"## {item.number}. {args.title}\n{rest}"
        start = item_starts(documents[0])[groups[0].index(item)]
        documents[0] = (
            documents[0][:start] + changed + documents[0][start + len(item.text) :]
        )
    elif args.command == "set-status":
        changed = re.sub(
            r"^Status: .*?$",
            f"Status: {args.status}",
            item.text,
            count=1,
            flags=re.MULTILINE,
        )
        documents[0] = documents[0].replace(item.text, changed, 1)
    else:
        heading, rest = item.text.split("\n", 1)
        rest = re.sub(r"^Status: .*?\n", "", rest, count=1, flags=re.MULTILINE)
        archived = (
            f"{heading}\nClosed: {date.today().isoformat()} - {args.resolution}\n{rest}"
        )
        documents[0] = documents[0].replace(item.text, "", 1)
        documents[1] = append_item(documents[1], archived)
    return None


def run(args) -> None:
    writing = args.command in {"add", "set-status", "edit", "close"}
    # The directory must exist; a typo must not create a new notes tree.
    if not args.dir.exists():
        raise ValueError(f"directory does not exist: {args.dir}")
    with (args.dir / ".backlog.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if writing else fcntl.LOCK_SH)
        documents = [
            (args.dir / name).read_text(encoding="utf-8")
            if (args.dir / name).exists()
            else header
            for name, header in zip(NAMES, HEADERS, strict=True)
        ]
        groups = validate(documents)
        if writing:
            after = documents.copy()
            number = update(args, after, groups)
            replace_documents(args.dir, documents, after)
            if number is not None:
                print(number)
        elif args.command == "next":
            print(
                max((item.number for group in groups for item in group), default=0) + 1
            )
        elif args.command == "list":
            print("| Number | Status | Title |\n| --- | --- | --- |")
            for group in groups if args.all else groups[:1]:
                for item in group:
                    title = item.title.replace("|", r"\|")
                    print(f"| {item.number} | {item.status} | {title} |")
        else:
            print("OK")


def main() -> int:
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, OSError, UnicodeError) as error:
        print(f"backlog: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
