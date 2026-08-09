"""First-pass parser for OKF (Open Knowledge Format) markdown bundles."""

import argparse
import json
import os
import re

import yaml

FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)\Z", re.DOTALL)
LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+\.md(?:#[^)\s]*)?)\)")


def parse_okf_file(filepath: str) -> dict:
    """Parse one OKF file into path/frontmatter/body/links, or None on failure."""
    try:
        with open(filepath, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        print(f"warning: could not read {filepath}: {exc}")
        return None

    match = FRONTMATTER_RE.match(text)
    if not match:
        if os.path.basename(filepath) == "index.md":
            links = [{"text": t, "target": g} for t, g in LINK_RE.findall(text)]
            return {
                "path": filepath,
                "frontmatter": {},
                "body": text,
                "links": links,
                "is_index": True,
            }
        print(f"warning: no frontmatter in {filepath}")
        return None

    raw_frontmatter, body = match.group(1), match.group(2)
    try:
        frontmatter = yaml.safe_load(raw_frontmatter)
    except yaml.YAMLError as exc:
        print(f"warning: bad YAML in {filepath}: {exc}")
        return None

    if not isinstance(frontmatter, dict):
        print(f"warning: frontmatter in {filepath} is not a mapping")
        return None

    links = [{"text": t, "target": g} for t, g in LINK_RE.findall(body)]
    return {
        "path": filepath,
        "frontmatter": frontmatter,
        "body": body,
        "links": links,
    }


def _md_files(bundle_dir: str) -> list:
    """List every .md file under bundle_dir, recursively, in stable order."""
    found = []
    for root, _dirs, files in os.walk(bundle_dir):
        found.extend(os.path.join(root, f) for f in sorted(files) if f.endswith(".md"))
    return found


def parse_bundle(bundle_dir: str) -> list[dict]:
    """Parse every .md file under bundle_dir, dropping the ones that fail."""
    results = (parse_okf_file(p) for p in _md_files(bundle_dir))
    return [r for r in results if r is not None]


def main() -> None:
    """Parse a single OKF file, or a whole bundle with --bundle, and report it."""
    ap = argparse.ArgumentParser(description="Parse an OKF markdown file.")
    ap.add_argument("path", nargs="?", help="path to the .md file")
    ap.add_argument("--bundle", help="parse every .md file under this directory")
    args = ap.parse_args()

    if bool(args.path) == bool(args.bundle):
        ap.error("give exactly one of: a file path, or --bundle DIR")

    if args.bundle:
        paths = _md_files(args.bundle)
        results = parse_bundle(args.bundle)
        parsed = {r["path"] for r in results}
        skipped = [p for p in paths if p not in parsed]
        idx = sum(1 for p in skipped if os.path.basename(p) == "index.md")
        idx_ok = sum(1 for r in results if r.get("is_index"))
        print(
            f"{len(paths)} .md files found, {len(results)} parsed "
            f"({idx_ok} index, {len(results) - idx_ok} concepts), "
            f"{len(skipped)} skipped ({idx} index.md, {len(skipped) - idx} other)"
        )
        return

    result = parse_okf_file(args.path)
    if result is None:
        raise SystemExit(1)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
