"""
typo_eval.py
A small experiment, not part of the main runs: does spelling correction win back what a typo loses?

For each of the 50 keyword queries one deterministic typo is made in the longest word (random.Random(42): delete,
swap or substitute one inner letter). BM25 (the R1 setting) is run on (a) the clean query, (b) the typo query as typed,
(c) the typo query with its rare or missing words corrected by correct_stem(). The script reports mean nDCG@10 for
each, and how often the typo changed the stem of the word and how often the correction brought the original stem back.

Usage: python -m evaluation.typo_eval   (writes evaluation/typo_eval.json, changes no other file)
"""

import json
import os
import random
import string
import sys
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.controller import load_config
from engine.query_understanding import MIN_CORRECTION_DF, correct_stem, correctable
from engine.ranker import rank_bm25
from engine.search import SearchIndex
from evaluation.metrics import compute_ndcg_at_k
from evaluation.run_eval import DEPTH
from evaluation.topics import FIELDS, load_qrels, load_topics
from pipeline.tokenizer import query_terms, stem_term, tokenize_no_stem

OUTPUT_PATH = os.path.join("evaluation", "typo_eval.json")
SEED = 42
MIN_TYPO_WORD_LENGTH = 4


def make_typo(word, rng):
    """One typo at an inner letter: delete it, swap it with the next letter, or replace it."""
    operation = rng.choice(["delete", "swap", "substitute"])
    if operation == "swap":
        position = rng.randrange(1, len(word) - 2)
        return word[:position] + word[position + 1] + word[position] + word[position + 2:], operation
    position = rng.randrange(1, len(word) - 1)
    if operation == "delete":
        return word[:position] + word[position + 1:], operation
    replacement = rng.choice([letter for letter in string.ascii_lowercase if letter != word[position]])
    return word[:position] + replacement + word[position + 1:], operation


def typo_query(text, rng):
    """Returns (typo text, original word, typo word, operation); the text is unchanged if no word is long enough."""
    words = [w for w in tokenize_no_stem(text) if w.isalpha() and len(w) >= MIN_TYPO_WORD_LENGTH]
    if not words:
        return text, None, None, None
    longest = max(words, key=len)
    typo, operation = make_typo(longest, rng)
    return text.lower().replace(longest, typo, 1), longest, typo, operation


def corrected_terms(text, index):
    terms = []
    for stem in query_terms(text):
        if index.get_df(stem) < MIN_CORRECTION_DF and correctable(stem):
            stem = correct_stem(stem, index) or stem
        terms.append(stem)
    return terms


def ndcg(terms, topic_id, index, config, qrels):
    ranked = rank_bm25(terms, index, top_k=DEPTH, k1=config["bm25_k1"], b=config["bm25_b"], title_boost=0.0)
    return compute_ndcg_at_k([index.get_ext_id(doc_id) for _, doc_id in ranked], qrels[topic_id], k=10)


def main():
    config, index = load_config(), SearchIndex()
    topics, qrels = load_topics(), load_qrels()
    getter = FIELDS["metadata.query"]
    rng = random.Random(SEED)
    rows = []
    for topic in topics:
        clean = getter(topic)
        typo_text, original, typo, operation = typo_query(clean, rng)
        record = {
            "topic": topic["_id"], "clean": clean, "typo_query": typo_text, "original_word": original, "typo_word": typo, "operation": operation,
            "ndcg_clean": ndcg(query_terms(clean), topic["_id"], index, config, qrels),
            "ndcg_typo": ndcg(query_terms(typo_text), topic["_id"], index, config, qrels),
            "ndcg_corrected": ndcg(corrected_terms(typo_text, index), topic["_id"], index, config, qrels),
        }
        if original:
            record["stem_changed"] = stem_term(typo) != stem_term(original)
            record["fixed_back"] = record["stem_changed"] and stem_term(original) in corrected_terms(typo_text, index)
        rows.append(record)
    typos = [r for r in rows if r["original_word"]]
    changed = [r for r in typos if r["stem_changed"]]
    summary = {
        "queries": len(rows), "typos_made": len(typos), "typos_that_changed_the_stem": len(changed),
        "fixed_back_to_the_original_stem": sum(r["fixed_back"] for r in changed),
        "mean_ndcg_at_10_clean": mean(r["ndcg_clean"] for r in rows),
        "mean_ndcg_at_10_typo": mean(r["ndcg_typo"] for r in rows),
        "mean_ndcg_at_10_typo_corrected": mean(r["ndcg_corrected"] for r in rows),
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"note": "diagnostic only: one deterministic typo (random.Random(42)) in the longest word of each of the 50 keyword queries, BM25 in the R1 setting",
                   "summary": summary, "queries": rows}, f, indent=2)
    for name, value in summary.items():
        print(f"{name}: {round(value, 4) if isinstance(value, float) else value}")


if __name__ == "__main__":
    main()
