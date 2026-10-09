"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REF_LINE = re.compile(r"^\[(\d+)\]\s*(.*)$")
_URL_PATTERN = re.compile(r"https?://[^\s)\]]+")


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not isinstance(sources, list) or len(sources) == 0:
        return ["no sources in sources.json"]

    seen_urls = set()
    seen_ns = set()
    source_by_n = {}
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"invalid source entry (not a dict): {entry!r}")
            continue
        n = entry.get("n")
        if type(n) is not int:
            problems.append(f"source entry has non-integer n: {n!r}")
        elif n in seen_ns:
            problems.append(f"duplicate source number n={n} in sources.json")
        else:
            seen_ns.add(n)
            source_by_n[n] = entry

        url = entry.get("url")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source n={n} has invalid url: {url!r}")
        elif url in seen_urls:
            problems.append(f"duplicate url in sources.json: {url}")
        else:
            seen_urls.add(url)

    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing '## References' section")
        return problems

    body = report_text[:matches[-1].start()]
    ref_section = report_text[matches[-1].end():]

    cited_in_body = set()
    segments = _CODE.split(body)
    for i, segment in enumerate(segments):
        if i % 2 == 1:
            continue
        for m in _GROUP.finditer(segment):
            group_str = m.group(1)
            for part in re.split(r"\s*,\s*", group_str):
                span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
                if span:
                    a, b = int(span.group(1)), int(span.group(2))
                    if 0 <= b - a <= 200:
                        cited_in_body.update(range(a, b + 1))
                    else:
                        cited_in_body.update([a, b])
                elif part.isdigit():
                    cited_in_body.add(int(part))

    for c in sorted(cited_in_body):
        if c not in source_by_n:
            problems.append(f"[{c}] cited but missing from sources.json")

    for n in sorted(source_by_n.keys()):
        if n not in cited_in_body:
            problems.append(f"source [{n}] never cited")

    ref_counts = {}
    for line in ref_section.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        m = _REF_LINE.match(line_str)
        if m:
            ref_n = int(m.group(1))
            ref_counts[ref_n] = ref_counts.get(ref_n, 0) + 1
            if ref_n not in source_by_n:
                problems.append(f"reference [{ref_n}] is not in sources.json")
            urls = _URL_PATTERN.findall(line_str)
            if len(urls) == 0:
                problems.append(f"reference [{ref_n}] has no URL")
            elif len(urls) > 1:
                problems.append(f"reference [{ref_n}] bundles multiple URLs")
            else:
                if ref_n in source_by_n and urls[0] != source_by_n[ref_n].get("url"):
                    problems.append(f"reference [{ref_n}] URL does not match sources.json ({urls[0]} != {source_by_n[ref_n].get('url')})")

    for n in sorted(source_by_n.keys()):
        count = ref_counts.get(n, 0)
        if count == 0:
            problems.append(f"missing reference line for source [{n}]")
        elif count > 1:
            problems.append(f"duplicate reference line for source [{n}] ({count} lines)")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
