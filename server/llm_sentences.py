"""
LLM-powered sentence completion — generation & validation via LM Studio.

Uses the OpenAI-compatible API exposed by LM Studio on localhost:1234.
Falls back gracefully when LM Studio is unavailable.
"""

import json
import random
import threading
import time
import traceback
from collections import deque

from openai import OpenAI

# ---------------------------------------------------------------------------
# LM Studio client
# ---------------------------------------------------------------------------
_LM_BASE_URL = "http://localhost:1234/v1"
# LM Studio local server does not require a real API key
_LM_API_KEY = "lm-studio"
_LM_MODEL = "qwen2.5-coder-14b-instruct"

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=_LM_BASE_URL, api_key=_LM_API_KEY)
    return _client


def is_llm_available() -> bool:
    """Check if LM Studio server is reachable and a model is loaded."""
    try:
        c = _get_client()
        c.models.list()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Proverb": "well-known English proverbs",
    "Idiom": "common English idioms and expressions",
    "Nursery Rhyme": "popular English nursery rhymes and children's songs",
    "Science": "interesting science and nature facts",
    "Geography": "world geography and country facts",
    "History": "notable historical events and facts",
    "Pop Culture": "popular movies, music, books, and TV references",
}

# ---------------------------------------------------------------------------
# Sentence pool — pre-generated, refilled in background
# ---------------------------------------------------------------------------
_POOL_TARGET = 15  # sentences to maintain per category
_POOL_REFILL = 5   # when pool drops below this, trigger refill
_BATCH_SIZE = 10   # sentences per LLM call

_pool: dict[str, deque] = {cat: deque() for cat in CATEGORIES}
_pool_lock = threading.Lock()
_next_id = 1000  # start IDs above hardcoded range (1-80)
_id_lock = threading.Lock()
_refill_in_progress: set[str] = set()


def _next_sentence_id() -> int:
    global _next_id
    with _id_lock:
        sid = _next_id
        _next_id += 1
        return sid


# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------
_GEN_PROMPT = """\
Generate {count} unique fill-in-the-blank sentences about {description}.

Rules:
- Each sentence MUST have exactly ONE blank shown as three underscores: ___
- The ___ replaces exactly ONE missing word (the answer)
- The missing word MUST be the KEY word that completes the meaning
- The sentence should make NO sense without the missing word
- The missing word must be clearly guessable from context
- Sentences should be varied in difficulty (some easy, some medium)
- Be creative — do NOT reuse famous proverbs or clichés
- Each answer must be a single common English word (no numbers)
- Place ___ where the key word belongs — NOT at the very end of a list

Example format:
[{{"text":"Water freezes at zero degrees ___.","answer":"celsius"}},
 {{"text":"The ___ is the closest star to Earth.","answer":"sun"}}]

Return ONLY a JSON array, no markdown fences, no explanation:
"""


def _generate_batch(category: str, count: int = _BATCH_SIZE) -> list[dict]:
    """Call LM Studio to generate a batch of sentences. Returns list of dicts."""
    desc = CATEGORIES.get(category, category)
    prompt = _GEN_PROMPT.format(count=count, description=desc)

    try:
        resp = _get_client().chat.completions.create(
            model=_LM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip markdown fences if present
        if "```" in raw:
            import re as _re
            m = _re.search(r"```(?:json)?\s*\n?(.*?)```", raw, _re.DOTALL)
            if m:
                raw = m.group(1).strip()

        # Extract JSON array even if there's surrounding text
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        raw = raw[start:end + 1]

        sentences = json.loads(raw)
        if not isinstance(sentences, list):
            return []

        result = []
        for s in sentences:
            text = s.get("text", "")
            answer = s.get("answer", "")
            if "___" in text and answer and isinstance(answer, str):
                result.append({
                    "id": _next_sentence_id(),
                    "cat": category,
                    "text": text.strip(),
                    "answers": [answer.strip().lower()],
                    "llm_generated": True,
                })
        return result
    except json.JSONDecodeError:
        # Try to salvage partial JSON by fixing common issues
        try:
            # Sometimes LLM outputs trailing comma or unescaped chars
            raw_fixed = raw.rstrip().rstrip(",") + "]" if not raw.rstrip().endswith("]") else raw
            sentences = json.loads(raw_fixed)
            result = []
            for s in sentences:
                text = s.get("text", "")
                answer = s.get("answer", "")
                if "___" in text and answer and isinstance(answer, str):
                    result.append({
                        "id": _next_sentence_id(),
                        "cat": category,
                        "text": text.strip(),
                        "answers": [answer.strip().lower()],
                        "llm_generated": True,
                    })
            return result
        except Exception:
            traceback.print_exc()
            return []
    except Exception:
        traceback.print_exc()
        return []


# ---------------------------------------------------------------------------
# Answer validation via LLM — with semantic feedback
# ---------------------------------------------------------------------------
_CHECK_PROMPT = """\
Sentence: "{sentence}"
Correct answer: "{answer}"
User said: "{spoken}"

Judge the user's answer and return ONLY a JSON object (no markdown, no extra text):
{{
  "correct": true or false,
  "closeness": "exact" or "synonym" or "close" or "wrong",
  "explanation": "One short sentence explaining WHY the answer is right or wrong, and what the sentence means."
}}

Guidelines:
- "exact": user said the correct word or a trivial variant (plural/singular, tense change). Mark as correct.
- "synonym": user said a valid synonym that fits the sentence well (e.g. "rock" for "stone", "birds" for "bird"). Mark as correct.
- "close": user's answer is semantically related but doesn't really fit (e.g. "robin" for "bird"). Mark as NOT correct.
- "wrong": completely unrelated answer. Mark as NOT correct.
- Be LENIENT: plurals, synonyms, and near-equivalents should be accepted (correct=true).
- Keep explanation under 25 words. Be encouraging even when wrong.
- If correct, explain what the completed sentence means or where it comes from.
- If wrong, briefly explain why the correct answer fits better."""


def check_answer_llm(sentence_text: str, spoken: str, canonical_answer: str) -> dict | None:
    """
    Use LLM to semantically judge if the spoken word is correct.
    Returns dict with keys: correct, closeness, explanation.
    Returns None if LLM is unavailable (caller should fall back).
    """
    if not spoken or not spoken.strip():
        return {"correct": False, "closeness": "wrong", "explanation": "No answer detected."}
    try:
        prompt = _CHECK_PROMPT.format(
            sentence=sentence_text,
            answer=canonical_answer,
            spoken=spoken.strip(),
        )
        resp = _get_client().chat.completions.create(
            model=_LM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip markdown fences if present
        if "```" in raw:
            import re as _re
            m = _re.search(r"```(?:json)?\s*\n?(.*?)```", raw, _re.DOTALL)
            if m:
                raw = m.group(1).strip()

        # Find JSON object
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            raw = raw[start:end + 1]

        result = json.loads(raw)
        return {
            "correct": bool(result.get("correct", False)),
            "closeness": str(result.get("closeness", "wrong")),
            "explanation": str(result.get("explanation", "")),
        }
    except json.JSONDecodeError:
        # Fallback: try to extract YES/NO from raw text
        try:
            upper = raw.upper()
            if "YES" in upper or '"correct": true' in raw.lower() or '"correct":true' in raw.lower():
                return {"correct": True, "closeness": "exact", "explanation": ""}
            if "NO" in upper or '"correct": false' in raw.lower() or '"correct":false' in raw.lower():
                return {"correct": False, "closeness": "wrong", "explanation": ""}
        except Exception:
            pass
        return None
    except Exception:
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------
def get_sentences(category: str | None = None, count: int = 10) -> list[dict]:
    """
    Return sentences from the pool. If category is None or 'all', mix from all.
    Each call pops from the front and triggers refill if needed.
    """
    with _pool_lock:
        if category and category != "all" and category in _pool:
            cats = [category]
        else:
            cats = list(CATEGORIES.keys())

        result = []
        for cat in cats:
            q = _pool[cat]
            take = min(len(q), count if len(cats) == 1 else max(2, count // len(cats)))
            for _ in range(take):
                if q:
                    result.append(q.popleft())

        # Trigger background refill for depleted categories
        for cat in cats:
            if len(_pool[cat]) < _POOL_REFILL and cat not in _refill_in_progress:
                _trigger_refill(cat)

    random.shuffle(result)
    return result


def get_pool_status() -> dict:
    """Return pool sizes per category."""
    with _pool_lock:
        return {cat: len(q) for cat, q in _pool.items()}


def _refill_category(category: str):
    """Refill a single category (runs in background thread)."""
    try:
        batch = _generate_batch(category, _BATCH_SIZE)
        if batch:
            with _pool_lock:
                _pool[category].extend(batch)
            print(f"[LLM] Refilled {category}: +{len(batch)} sentences "
                  f"(pool now {len(_pool[category])})")
    except Exception:
        traceback.print_exc()
    finally:
        _refill_in_progress.discard(category)


def _trigger_refill(category: str):
    """Start a background thread to refill a category."""
    _refill_in_progress.add(category)
    t = threading.Thread(target=_refill_category, args=(category,), daemon=True)
    t.start()


def prefill_all():
    """
    Fill all category pools at startup. Called once.
    Runs generation sequentially to avoid overloading LM Studio.
    """
    print("[LLM] Pre-filling sentence pools…")
    for cat in CATEGORIES:
        batch = _generate_batch(cat, _POOL_TARGET)
        if batch:
            with _pool_lock:
                _pool[cat].extend(batch)
            print(f"  {cat}: {len(batch)} sentences")
        else:
            print(f"  {cat}: FAILED (LLM unavailable?)")
    total = sum(len(q) for q in _pool.values())
    print(f"[LLM] Pre-fill complete: {total} sentences across {len(CATEGORIES)} categories")


def prefill_async():
    """Start prefill in a background thread so server starts immediately."""
    t = threading.Thread(target=prefill_all, daemon=True)
    t.start()
