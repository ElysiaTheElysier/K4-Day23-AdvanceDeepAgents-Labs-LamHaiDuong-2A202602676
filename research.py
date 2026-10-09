"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, NOTES_DIR, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    clean = re.sub(r"[^\w]+", "-", (topic or "").strip().lower()).strip("-")
    clean = clean[:60].strip("-")
    return clean or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Conduct a deep, rigorous research survey on the topic: '{topic}'.\n\n"
        f"MANDATORY REQUIREMENTS:\n"
        f"1. You MUST call `write_todos` to create an actionable research plan.\n"
        f"2. You MUST invoke the `task` tool at least 3 to 4 times (with `subagent_type='researcher'`) to delegate research on separate sub-questions in parallel.\n"
        f"   To guarantee multi-source diversity across all 3 families:\n"
        f"   - Task 1 description MUST tell researcher: 'Search using hf_search_papers for core architectures and methods (source: hf-search)'.\n"
        f"   - Task 2 description MUST tell researcher: 'Search using hf_daily_papers for trending advances and models (source: hf-daily)'.\n"
        f"   - Task 3 description MUST tell researcher: 'Search using arxiv_search for academic foundations and benchmarks (source: arxiv)'.\n"
        f"   Notes MUST be saved to {NOTES_DIR}/<NN>-<slug>.md.\n"
        f"3. Read researcher notes in {NOTES_DIR} and consolidate all retrieved sources into {SOURCES_PATH}. "
        f"CRITICAL: The sources in {SOURCES_PATH} MUST include at least 3 distinct source families among arxiv, hf-daily, hf-search, and web (specifically ensure 'arxiv', 'hf-daily', and 'hf-search' are all present in {SOURCES_PATH} and cited in the report!).\n"
        f"4. Write the comprehensive synthesized survey report into {REPORT_PATH} following REPORT_TEMPLATE.md. "
        f"CRITICAL CITATION REQUIREMENT: You MUST include bracketed inline citations [n] throughout every section for EVERY model, method, or paper mentioned. "
        f"You MUST cite at least 6 to 10 distinct sources in the report body, including sources from 'arxiv', 'hf-daily', AND 'hf-search'. "
        f"Do NOT leave any source family uncited, because {FINALIZER_PATH} will automatically discard any uncited sources from {SOURCES_PATH}! "
        f"Do not write the References section yourself.\n"
        f"5. Run {FINALIZER_PATH} via `execute` (no arguments) to generate ## References and clean up sources.json.\n"
        f"6. Run {VALIDATOR_PATH} via `execute` (`python3 {VALIDATOR_PATH}`) to verify all citations resolve cleanly.\n"
        f"7. Invoke `task` with `subagent_type='citation-checker'` to spot-check key claims against source URLs.\n"
        f"Ensure all files ({REPORT_PATH} and {SOURCES_PATH}) are finalized and consistent."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}."""
    tool_calls_counter = Counter()
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        # Tool calls
        tcs = getattr(msg, "tool_calls", None)
        if not tcs and isinstance(msg, dict):
            tcs = msg.get("tool_calls")
        if tcs:
            for tc in tcs:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name:
                    tool_calls_counter[name] += 1

        # Usage metadata
        meta = getattr(msg, "usage_metadata", None)
        if not meta and isinstance(msg, dict):
            meta = msg.get("usage_metadata")
        if meta and isinstance(meta, dict):
            input_tokens += meta.get("input_tokens", 0) or 0
            output_tokens += meta.get("output_tokens", 0) or 0

    subagent_calls = tool_calls_counter.get("task", 0)
    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_calls_counter),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError(f"Report at {REPORT_PATH} is missing or empty")
    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError(f"Sources at {SOURCES_PATH} is missing or empty")

    try:
        sources_data = json.loads(sources_bytes.decode("utf-8"))
        if not isinstance(sources_data, list):
            raise ValueError("sources.json is not a JSON list")
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON in {SOURCES_PATH}: {exc}")

    slug = slugify(topic)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    source_families = sorted(list({s.get("source") for s in sources_data if isinstance(s, dict) and s.get("source")}))
    n_sources = len(sources_data)

    stats = summarize(messages, elapsed, model_name)
    meta = {
        "topic": topic,
        **stats,
        "n_sources": n_sources,
        "source_families": source_families,
    }

    md_path = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"

    md_path.write_bytes(report_bytes)
    sources_path.write_bytes(sources_bytes)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return md_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    clean_topic = (topic or "").strip()
    if not clean_topic:
        print("Usage: python research.py '<topic>'", file=sys.stderr)
        return 2

    model = make_model()
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL") or "unknown"
    start = time.monotonic()

    with open_sandbox() as backend:
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(
            backend,
            {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            },
        )
        agent = build_lead_agent(backend, model)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": build_prompt(clean_topic)}]},
            config={"recursion_limit": 1000},
        )
        elapsed = time.monotonic() - start
        messages = result.get("messages", []) if isinstance(result, dict) else []

        try:
            saved_path = save_outputs(backend, clean_topic, messages, elapsed, model_name)
        except RuntimeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1

    print(f"Report saved to {saved_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
