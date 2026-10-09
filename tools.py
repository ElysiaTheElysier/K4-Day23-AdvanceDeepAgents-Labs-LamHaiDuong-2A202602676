"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import time  # noqa: F401
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError or transient network errors, wait and call it again."""
    import random

    for attempt in range(attempts):
        try:
            return fn()
        except (RetryableError, httpx.HTTPStatusError, httpx.TransportError) as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                if exc.response.status_code not in (429, 500, 502, 503, 504):
                    raise
            if attempt == attempts - 1:
                raise

            retry_after = None
            if isinstance(exc, RetryableError):
                retry_after = exc.retry_after
            elif isinstance(exc, httpx.HTTPStatusError):
                ra = exc.response.headers.get("Retry-After")
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        pass

            if retry_after is not None and retry_after > 0:
                delay = min(retry_after, cap)
            else:
                jitter = random.uniform(0.1, 1.0)
                delay = min(base * (2 ** attempt) + jitter, cap)

            time.sleep(delay)


# ---- TODO 2: arXiv ----
_LAST_ARXIV_CALL = 0.0


def _arxiv_rate_limit_wait():
    global _LAST_ARXIV_CALL
    now = time.monotonic()
    elapsed = now - _LAST_ARXIV_CALL
    if elapsed < 3.0:
        time.sleep(3.0 - elapsed)
    _LAST_ARXIV_CALL = time.monotonic()


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    try:
        import re
        import xml.etree.ElementTree as ET

        terms = re.findall(r"[\w-]+", query)
        if not terms:
            return "NO RESULTS"

        search_query = " AND ".join(f"all:{t}" for t in terms)
        clamped_results = max(1, min(max_results, 30))

        def _call():
            _arxiv_rate_limit_wait()
            resp = httpx.get(
                ARXIV_URL,
                params={
                    "search_query": search_query,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "max_results": clamped_results,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.text

        xml_text = with_retry(_call, attempts=3, base=1.0, cap=15.0)
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "NO RESULTS"

        records = []
        for entry in entries:
            raw_id = entry.findtext("atom:id", default="", namespaces=ns)
            paper_id = raw_id.split("/abs/")[-1].strip() if "/abs/" in raw_id else raw_id.strip()
            paper_id = re.sub(r"v\d+$", "", paper_id)
            url = f"https://arxiv.org/abs/{paper_id}"
            published = entry.findtext("atom:published", default="", namespaces=ns)[:10]
            raw_title = entry.findtext("atom:title", default="", namespaces=ns)
            title = re.sub(r"\s+", " ", raw_title).strip()
            raw_summary = entry.findtext("atom:summary", default="", namespaces=ns)
            summary = re.sub(r"\s+", " ", raw_summary).strip()
            if len(summary) > 600:
                summary = summary[:600] + "..."
            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "source": "arxiv",
            })

        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        import re

        from datetime import datetime, timedelta, timezone

        clamped_limit = max(1, min(limit, 100))
        dates_to_query = [date.strip()] if date else [""] + [
            (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 5)
        ]

        data = []
        seen_ids = set()
        for d in dates_to_query:
            p = {"limit": clamped_limit}
            if d:
                p["date"] = d
            try:
                def _call():
                    resp = httpx.get(HF_DAILY_URL, params=p, timeout=20.0)
                    resp.raise_for_status()
                    return resp.json()

                batch = with_retry(_call, attempts=3, base=1.0, cap=15.0)
                if isinstance(batch, list):
                    for item in batch:
                        paper = item.get("paper") if isinstance(item, dict) and isinstance(item.get("paper"), dict) else item
                        pid = paper.get("id") if isinstance(paper, dict) else None
                        if pid and pid not in seen_ids:
                            seen_ids.add(pid)
                            data.append(item)
            except Exception:
                continue

        if not data:
            return "NO RESULTS"

        records = []
        kw = keyword.strip().lower() if keyword else ""
        words = [w for w in re.findall(r"\w+", kw) if len(w) > 3]
        all_candidates = []
        for item in data:
            paper = item.get("paper") if isinstance(item, dict) and isinstance(item.get("paper"), dict) else item
            if not isinstance(paper, dict) or not paper.get("id"):
                continue
            paper_id = str(paper["id"])
            url = f"https://huggingface.co/papers/{paper_id}"
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            title = re.sub(r"\s+", " ", str(paper.get("title") or item.get("title") or "")).strip()
            summary = re.sub(r"\s+", " ", str(paper.get("summary") or item.get("summary") or "")).strip()
            if len(summary) > 600:
                summary = summary[:600] + "..."
            upvotes = int(paper.get("upvotes") or 0)
            github = str(paper.get("githubRepo") or "")
            stars = int(paper.get("githubStars") or 0)

            rec = {
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
                "source": "hf-daily",
            }
            all_candidates.append(rec)

            text = (title + " " + summary).lower()
            if kw and kw not in text and (not words or not any(w in text for w in words)):
                continue

            records.append(rec)

        if not records and all_candidates:
            records = all_candidates[:clamped_limit]

        records.sort(key=lambda r: r.get("upvotes", 0), reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        import re

        clamped_limit = max(1, min(limit, 50))
        params = {"q": query.strip(), "limit": clamped_limit}

        def _call():
            resp = httpx.get(HF_SEARCH_URL, params=params, timeout=30.0)
            resp.raise_for_status()
            return resp.json()

        data = with_retry(_call, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list) or not data:
            return "NO RESULTS"

        records = []
        for item in data:
            paper = item.get("paper") if isinstance(item, dict) and isinstance(item.get("paper"), dict) else item
            if not isinstance(paper, dict) or not paper.get("id"):
                continue
            paper_id = str(paper["id"])
            url = f"https://huggingface.co/papers/{paper_id}"
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            title = re.sub(r"\s+", " ", str(paper.get("title") or item.get("title") or "")).strip()
            raw_summary = paper.get("ai_summary") or paper.get("summary") or item.get("summary") or ""
            summary = re.sub(r"\s+", " ", str(raw_summary)).strip()
            if len(summary) > 600:
                summary = summary[:600] + "..."
            upvotes = int(paper.get("upvotes") or 0)
            github = str(paper.get("githubRepo") or "")
            stars = int(paper.get("githubStars") or 0)

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
                "source": "hf-search",
            })

        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _redact_key(text: str, key: str) -> str:
    if key and key in text:
        return text.replace(key, "[REDACTED]")
    return text


def _call_exa_mcp(tool_name: str, arguments: dict) -> str:
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    url = EXA_URL
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    params = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        params["exaApiKey"] = api_key

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    def _call():
        resp = httpx.post(url, headers=headers, params=params, json=payload, timeout=45.0)
        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            if ra and ra.isdigit() and float(ra) > 60:
                raise RuntimeError(f"Exa rate limit quota reached (Retry-After: {ra}s)")
            delay = float(ra) if ra and ra.isdigit() else 20.0
            raise RetryableError("Exa HTTP 429 rate limited", retry_after=delay)
        resp.raise_for_status()

        raw_text = resp.text.strip()
        data = None
        if raw_text.startswith("data:"):
            for line in raw_text.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    line_data = line[5:].strip()
                    if line_data:
                        try:
                            data = json.loads(line_data)
                            break
                        except json.JSONDecodeError:
                            continue
        else:
            try:
                data = resp.json()
            except Exception:
                pass

        if not data:
            return ""

        if "error" in data:
            err = data["error"]
            err_msg = str(err.get("message", err))
            if "rate limit" in err_msg.lower() or "429" in err_msg or "too many requests" in err_msg.lower():
                raise RetryableError(f"Exa rate limited: {err_msg}", retry_after=20.0)
            raise RuntimeError(f"Exa error: {err_msg}")

        result = data.get("result", {})
        meta = result.get("_meta", {})
        meta_str = json.dumps(meta).lower()
        if "ratelimit" in meta_str or "rate_limit" in meta_str or "limited" in meta_str:
            raise RetryableError("Exa rate limited per _meta flag", retry_after=20.0)

        content = result.get("content", [])
        texts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                text_val = c.get("text", "")
                if "rate limit" in text_val.lower() and "exceeded" in text_val.lower():
                    raise RetryableError("Exa rate limited in content text", retry_after=20.0)
                texts.append(text_val)

        return "\n\n".join(texts).strip()

    return with_retry(_call, attempts=2, base=2.0, cap=20.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    try:
        obj = objective.strip() if objective else f"find comprehensive technical and research sources on {query}"
        results = _call_exa_mcp(
            "web_search_exa",
            {"query": query.strip(), "objective": obj, "numResults": max(1, min(num_results, 10))},
        )
        return results if results else "NO RESULTS"
    except Exception as exc:
        msg = f"ERROR: {type(exc).__name__}: {exc}"
        return _redact_key(msg, api_key)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    try:
        content = _call_exa_mcp("web_fetch_exa", {"urls": [url.strip()]})
        if not content:
            return "NO RESULTS"
        if len(content) > 12000:
            content = content[:12000] + "\n\n...[content truncated]"
        return content
    except Exception as exc:
        msg = f"ERROR: {type(exc).__name__}: {exc}"
        return _redact_key(msg, api_key)


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n", flush=True)
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n", flush=True)
