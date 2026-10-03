"""Label a fixed random sample of 1,500 reviews with a local LLM (Ollama, llama3.2).

One review per request, with a JSON schema that forces a yes/no answer for every complaint
theme. Results are cached one line per review in data/llm_cache.jsonl, so a rerun only sends
reviews that are not cached yet. If Ollama is unreachable, a keyword fallback labels the
sample instead and every row is tagged method='keyword'.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

SAMPLE_N, SEED = 1500, 42
CACHE = "data/llm_cache.jsonl"
OLLAMA = "http://localhost:11434"
THEMES = {
    "fit": "runs small/large, had to size up/down, too tight/loose/long/short, boxy (true even if fixed by exchanging sizes)",
    "fabric": "fabric thin, see-through, itchy, scratchy, cheap feel, stiff, wrinkles, pills",
    "build": "bad stitching/seams, holes, tears, shrinks, falls apart, broken zipper/buttons",
    "photo": "color, pattern or look differs from the photo/website",
    "style": "cut or design unflattering, frumpy, odd shape, looks like maternity wear",
    "price": "overpriced, not worth the money, only buy on sale",
}
SCHEMA = {"type": "object", "required": ["tone", *THEMES],
          "properties": {"tone": {"type": "string", "enum": ["pos", "neg", "mixed"]},
                         **{k: {"type": "boolean"} for k in THEMES}}}
# Fixed instructions first and the review last, so Ollama reuses the cached prompt prefix.
PROMPT = ("Label one women's clothing review. For each complaint key, answer true only if the reviewer "
          "complains about it for THIS item:\n"
          + "\n".join(f"{k}: {v}" for k, v in THEMES.items())
          + "\ntone: overall sentiment (pos, neg or mixed).\nReview: ")

# Fallback keyword lists, used only when Ollama is not reachable.
KEYWORDS = {
    "fit": ["runs small", "runs large", "runs big", "too small", "too big", "too tight", "too loose", "too short", "too long", "boxy", "size down", "size up", "sized down", "sized up"],
    "fabric": ["thin", "see through", "see-through", "itchy", "scratchy", "cheap", "pill", "wrinkl", "stiff"],
    "build": ["seam", "stitch", "hole", "tore", "ripped", "shrank", "shrunk", "fell apart", "unravel"],
    "photo": ["than pictured", "in the picture", "in the photo", "different color", "looks nothing like"],
    "style": ["unflattering", "frumpy", "maternity", "shapeless"],
    "price": ["overpriced", "not worth", "expensive", "on sale"],
}


def llm_label(text):
    r = requests.post(f"{OLLAMA}/api/generate", timeout=300, json={
        "model": "llama3.2", "prompt": PROMPT + text, "format": SCHEMA, "stream": False,
        "options": {"temperature": 0}})
    return {**json.loads(r.json()["response"]), "method": "llm"}


def keyword_label(text):
    t = text.lower()
    return {"tone": None, **{k: any(w in t for w in words) for k, words in KEYWORDS.items()}, "method": "keyword"}


reviews = pd.read_parquet("data/clean/reviews.parquet")
sample = reviews.sample(SAMPLE_N, random_state=SEED)["text"]
done = set()
if os.path.exists(CACHE):
    # only LLM rows count as done, so keyword-fallback rows get relabelled once Ollama is back
    done = {r["review_id"] for r in map(json.loads, open(CACHE, encoding="utf-8")) if r["method"] == "llm"}
todo = sample[~sample.index.isin(done)]
print(f"{len(done)} cached, {len(todo)} to label")

label = llm_label
try:
    if len(todo):
        requests.get(OLLAMA, timeout=5)
except requests.RequestException:
    label = keyword_label
    print("Ollama not reachable: using keyword fallback")

def safe_label(text):
    try:
        return label(text)
    except (ValueError, KeyError, requests.RequestException):  # bad output or failed call: leave uncached, a rerun retries it
        return None


# Batches of 4 concurrent requests: Ollama serves them in parallel slots, ~2x the
# throughput of one-at-a-time on this CPU-only machine.
with open(CACHE, "a", encoding="utf-8") as f, ThreadPoolExecutor(4) as pool:
    for n, (i, row) in enumerate(zip(todo.index, pool.map(safe_label, todo)), 1):
        if row:
            f.write(json.dumps({"review_id": i, **row}) + "\n")
            f.flush()
        if n % 25 == 0:
            print(f"{n}/{len(todo)}", flush=True)

labels = pd.read_json(CACHE, lines=True).drop_duplicates("review_id", keep="last")
labels = labels[labels["review_id"].isin(sample.index)]
labels.merge(reviews, left_on="review_id", right_index=True).to_parquet("data/clean/labels.parquet")
print(f"{len(labels)} of {SAMPLE_N} sample reviews labelled")
