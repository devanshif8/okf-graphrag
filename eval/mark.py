"""Mark the judgement sheet one row at a time, from the terminal.

`judge.py pool` writes 738 rows and a spreadsheet is a poor place to read 738
snippets: the eye drifts, the strictness drifts with it, and a sheet judged
loosely in its second half rebuilds the saturated test the hand-judging exists
to escape. This shows one chapter at a time, asks one question, and writes the
answer straight back to the sheet.

    python eval/mark.py library/judgements.csv
    python eval/mark.py library/judgements.csv --needs 15
    python eval/mark.py library/probe_judgements.csv --all      # revisit marks

Then, exactly as from the spreadsheet:

    python eval/judge.py build library/judgements.csv \\
        --queries eval/need_queries.jsonl --output eval/judged_queries.jsonl --partial
    python eval/evaluate.py eval/judged_queries.jsonl --group-by kind

The question is always **does this chapter answer the need**, not is it on the
same subject. Shelf-level relevance already scores 0.83 and saturates; a
generous 1 here buys nothing back.

Three properties matter more than the ergonomics:

- **Every keystroke is saved.** The sheet is rewritten atomically after each
  mark, so quitting, a closed terminal or a dead laptop costs nothing. Marking
  is resumable across sessions because the sheet *is* the state.
- **The rows are shown in file order.** `pool` already shuffled them within each
  need to hide which retriever ranked what, and reordering here — by title, by
  book, anything — would hand back exactly the hint that shuffle removed.
- **Nothing is inferred.** A skipped row stays blank rather than becoming a 0,
  because `judge.py build` drops unjudged needs and counts blanks as irrelevant
  only when told to; a marker that quietly filled them in would turn an
  unfinished afternoon into a real-looking score.
"""

import argparse
import csv
import os
import shutil
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.judge import FIELDS, NO, YES  # noqa: E402

# Snippets carry em-dashes and accented titles out of Gutenberg, and a Windows
# console in cp1252 raises on them rather than printing a box. Judging must not
# die on a punctuation mark.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# What each key does. Derived from judge.py's own spellings so the two cannot
# drift apart, narrowed to the single characters a keypress can actually be.
MARK_KEYS = {
    **{key: "1" for key in YES if len(key) == 1},
    **{key: "0" for key in NO if len(key) == 1},
}
SKIP_KEYS = {"s", " ", "\r", "\n"}
UNDO_KEYS = {"u", "\x7f", "\b"}
# Escape is deliberately not a quit key: on a POSIX terminal an arrow key sends
# it as the first byte of a sequence, and losing the session to a stray arrow
# would be a poor reward for reaching for one.
QUIT_KEYS = {"q", "\x03", "\x04"}

WIDTH = 100


def load_rows(path: str) -> list[dict]:
    """Read the whole sheet, preserving row order and every column."""
    if not os.path.exists(path):
        raise SystemExit(f"no such sheet: {path} - run 'judge.py pool' first")
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise SystemExit(f"{path} is missing column(s): {', '.join(sorted(missing))}")
        return [{field: (row.get(field) or "") for field in FIELDS} for row in reader]


def save_rows(rows: list[dict], path: str) -> None:
    """Rewrite the sheet atomically, so an interrupted write cannot truncate it.

    The temporary file sits beside the sheet rather than in the system temp
    directory: `os.replace` is only atomic within one filesystem.
    """
    temporary = f"{path}.tmp"
    # utf-8-sig to match what `judge.py pool` wrote, so Excel still opens it.
    with open(temporary, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def pending(rows: list[dict], revisit: bool = False, needs: int | None = None) -> list[int]:
    """Which row indices to present, in sheet order.

    Unmarked rows only by default. `needs` caps how many distinct needs are
    offered, which is the honest way to stop early: 15 needs judged strictly
    beat 41 judged in a hurry, and `build` drops the untouched ones.
    """
    order: list[str] = []
    chosen = []
    for index, row in enumerate(rows):
        if not revisit and row["relevant"].strip():
            continue
        if row["need"] not in order:
            if needs is not None and len(order) >= needs:
                continue
            order.append(row["need"])
        chosen.append(index)
    return chosen


def tally(rows: list[dict]) -> dict:
    """Count what has been decided, for the progress line and the exit summary."""
    yes = sum(1 for row in rows if row["relevant"].strip().lower() in YES)
    no = sum(1 for row in rows if row["relevant"].strip().lower() in NO)
    return {"rows": len(rows), "yes": yes, "no": no, "blank": len(rows) - yes - no}


def render(row: dict, position: int, total: int, need_number: int, need_total: int) -> str:
    """One screenful: the need, the chapter, the text to judge it by.

    The domain label is absent by construction — the sheet never carried one —
    and so is any hint of which retriever found this chapter or at what rank.
    """
    width = min(shutil.get_terminal_size((WIDTH, 24)).columns, WIDTH)
    rule = "-" * width
    return "\n".join(
        [
            "",
            rule,
            f"need {need_number}/{need_total}   row {position}/{total}",
            "",
            textwrap.fill(row["need"], width),
            rule,
            f"  {row['title']}",
            "",
            textwrap.fill(row["snippet"], width - 2, initial_indent="  ", subsequent_indent="  "),
            rule,
        ]
    )


def has_console() -> bool:
    """Whether a keypress can actually arrive on stdin.

    **`sys.stdin.isatty()` is not that question on Windows.** Under Git Bash,
    MSYS and an agent's shell it answers True with no console attached — even
    for `< /dev/null` — and `msvcrt.getwch()` then blocks forever waiting for a
    key nobody can press. `GetConsoleMode` fails on a handle that is not a
    console, which is the distinction that matters here.
    """
    if os.name != "nt":
        return sys.stdin.isatty()

    import ctypes
    import msvcrt

    try:
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
    except OSError:  # io.UnsupportedOperation for a stdin with no descriptor
        return False
    mode = ctypes.c_uint()
    return bool(
        ctypes.windll.kernel32.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(mode))
    )


def require_terminal() -> None:
    """Refuse to start without a console, rather than hanging on one keypress.

    Without one the marker prints its first chapter and waits forever, which
    from the outside is indistinguishable from a judge reading slowly. It is
    cheaper to say so once, up front, than to lose an afternoon to it.
    """
    if not has_console():
        raise SystemExit(
            "mark.py needs a console - there is no keyboard attached to stdin here.\n"
            "Open a terminal in the project directory and run it there. Under an "
            "agent's shell, a pipe or CI it would wait for a keypress that cannot "
            "arrive, and mark nothing."
        )


def getkey() -> str:
    """Read one keypress, falling back to a line where cbreak is unavailable.

    `require_terminal` has already established there is a console, so the
    fallback here means an unusual one rather than a pipe, and an empty line
    reads as a skip.
    """
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if not line:
            return "q"
        return line.strip()[:1].lower() or "s"

    try:
        import msvcrt
    except ImportError:
        pass
    else:
        return msvcrt.getwch().lower()

    import termios
    import tty

    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)


def _prompt() -> str:
    return "  [1] answers it   [0] does not   [s] skip   [u] undo   [q] save and quit"


def _loop(rows: list[dict], queue: list[int], path: str) -> None:
    """Present each queued row until the queue empties or the judge quits.

    Undo pushes the row back onto the front of the queue *and* restores what the
    cell held before, so a mis-keyed 1 leaves no trace in the sheet.
    """
    needs = list(dict.fromkeys(rows[index]["need"] for index in queue))
    total = len(queue)
    history: list[tuple[int, str]] = []
    done = 0

    while queue:
        index = queue[0]
        row = rows[index]
        print(render(row, done + 1, total, needs.index(row["need"]) + 1, len(needs)))
        print(_prompt())

        key = getkey()
        if key in QUIT_KEYS:
            print("\n  stopping here - everything marked is already saved.")
            return
        if key in UNDO_KEYS:
            if not history:
                print("\n  nothing to undo yet.")
                continue
            previous_index, previous_value = history.pop()
            rows[previous_index]["relevant"] = previous_value
            save_rows(rows, path)
            queue.insert(0, previous_index)
            done = max(done - 1, 0)
            continue
        if key in SKIP_KEYS:
            queue.pop(0)
            done += 1
            continue
        if key not in MARK_KEYS:
            print(f"\n  '{key}' is not a key here.")
            continue

        history.append((index, row["relevant"]))
        row["relevant"] = MARK_KEYS[key]
        save_rows(rows, path)
        queue.pop(0)
        done += 1

    print("\n  that is every row in the queue.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sheet", help="the judgement sheet from judge.py pool")
    ap.add_argument(
        "--needs", type=int,
        help="offer only this many needs - 15 marked strictly is the advice",
    )
    ap.add_argument(
        "--all", dest="revisit", action="store_true",
        help="include rows already marked, to review or change them",
    )
    args = ap.parse_args()

    if args.needs is not None and args.needs < 1:
        raise SystemExit("--needs must be at least 1")

    require_terminal()

    rows = load_rows(args.sheet)
    if not rows:
        raise SystemExit(f"{args.sheet} has no rows to judge")

    before = tally(rows)
    queue = pending(rows, args.revisit, args.needs)
    if not queue:
        print(f"nothing left to mark in {args.sheet} - pass --all to review what is there")
        return

    print(f"{args.sheet}: {before['blank']} of {before['rows']} rows unmarked")
    print(f"marking {len(queue)} rows across "
          f"{len({rows[index]['need'] for index in queue})} needs")
    print("\njudge the text, not the subject: does this chapter answer the need?")

    try:
        _loop(rows, queue, args.sheet)
    except KeyboardInterrupt:
        print("\n\n  interrupted - everything marked is already saved.")

    after = tally(rows)
    print(f"\n  {after['yes']} relevant, {after['no']} not, {after['blank']} blank")
    print(f"  {before['blank'] - after['blank']} marked this session\n")
    print("  next:")
    print("    python eval/judge.py build", args.sheet,
          "--queries eval/need_queries.jsonl \\")
    print("        --output eval/judged_queries.jsonl --partial")
    print("    python eval/evaluate.py eval/judged_queries.jsonl --group-by kind")


if __name__ == "__main__":
    main()
