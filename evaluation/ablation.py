"""
ablation.py
A diagnostic, not part of the main runs: which move of the controller matters?

Each variant is R4 (controller v2.1, no clarification, evidence weight 0) with one thing switched off:
one move (config key disabled_moves) or index elimination (max_df_fraction = 1.0 for that run only; note that
this also lets feedback terms be common words, because feedback uses the same limit).
For every variant and field the script reports nDCG@10, P@10, the in-band rate and the mean number of hits, and a
paired randomization p-value against full R4 on nDCG@10. Nothing was changed because of these numbers.

Usage: python -m evaluation.ablation   (writes evaluation/ablation.json, changes no other file)
"""

import json
import os
import random
import sys
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.controller import load_config, run_controller
from engine.search import SearchIndex
from evaluation.run_eval import DEPTH, score_ranking, score_set
from evaluation.significance import ALPHA, SEED, compare
from evaluation.topics import FIELDS, load_qrels, load_topics

OUTPUT_PATH = os.path.join("evaluation", "ablation.json")
VARIANTS = {
    "none (full R4)": {},
    "no TIGHTEN_TITLE_ZONE": {"disabled_moves": ["TIGHTEN_TITLE_ZONE"]},
    "no TIGHTEN_ADD_PRF": {"disabled_moves": ["TIGHTEN_ADD_PRF"]},
    "no TIGHTEN_ADD_CONCEPT": {"disabled_moves": ["TIGHTEN_ADD_CONCEPT"]},
    "no RELAX_EXPAND_SYNONYMS": {"disabled_moves": ["RELAX_EXPAND_SYNONYMS"]},
    "no RELAX_WILDCARD": {"disabled_moves": ["RELAX_WILDCARD"]},
    "no RELAX_DROP_CONCEPT": {"disabled_moves": ["RELAX_DROP_CONCEPT"]},
    "no index elimination": {"max_df_fraction": 1.0},
}


def run_variant(overrides, field_getter, topics, qrels, index, base_config):
    config = dict(base_config, evidence_weight=0.0, **overrides)
    per_topic = {}
    for topic in topics:
        ranked, log, _ = run_controller(field_getter(topic), index, config, allow_clarify=False, top_k=DEPTH)
        ext_ids = [index.get_ext_id(doc_id) for _, doc_id in ranked]
        record = score_ranking(ext_ids, qrels[topic["_id"]])
        record.update(score_set({index.get_ext_id(d) for d in log.final_docs}, qrels[topic["_id"]], config))
        per_topic[topic["_id"]] = record
    return per_topic


def summarize(per_topic):
    return {
        "ndcg_at_10": mean(r["ndcg_at_10"] for r in per_topic.values()),
        "p_at_10": mean(r["p_at_10"] for r in per_topic.values()),
        "in_band_rate": mean(float(r["in_band"]) for r in per_topic.values()),
        "mean_hits": mean(r["hits"] for r in per_topic.values()),
    }


def main():
    base_config, index = load_config(), SearchIndex()
    topics, qrels = load_topics(), load_qrels()
    with open(os.path.join("evaluation", "results.json"), "r", encoding="utf-8") as f:
        saved_r4 = json.load(f)["runs"]["R4"]["fields"]
    rng = random.Random(SEED)
    output = {"note": "diagnostic only: each variant is R4 with one thing switched off; nothing was changed because of these results",
              "variants": {}, "tests_run": (len(VARIANTS) - 1) * len(FIELDS)}
    threshold = ALPHA / output["tests_run"]
    output["bonferroni_threshold"] = threshold
    for field, getter in FIELDS.items():
        full = None
        for name, overrides in VARIANTS.items():
            per_topic = run_variant(overrides, getter, topics, qrels, index, base_config)
            if full is None:
                full = per_topic
                assert all(abs(per_topic[t]["ndcg_at_10"] - saved_r4[field]["per_topic"][t]["ndcg_at_10"]) < 1e-12 for t in per_topic), "full R4 differs from results.json"
            row = summarize(per_topic)
            if overrides:
                pair = {"runs": {"R4": {"fields": {field: {"per_topic": full}}}, "VARIANT": {"fields": {field: {"per_topic": per_topic}}}}}
                test = compare(pair, "R4", "VARIANT", field, "ndcg_at_10", rng)
                row.update(diff_vs_r4=-test["mean_difference"], p_value=test["p_value"], variant_wins=test["topics_b_wins"],
                           r4_wins=test["topics_a_wins"], tied=test["topics_tied"], significant_after_bonferroni=test["p_value"] < threshold)
            output["variants"].setdefault(name, {})[field] = row
            print(f"{field:15} {name:26} nDCG@10={row['ndcg_at_10']:.4f} P@10={row['p_at_10']:.4f} in_band={row['in_band_rate']:.2f} "
                  f"hits={row['mean_hits']:.1f}" + (f" diff={row['diff_vs_r4']:+.4f} p={row['p_value']:.4f}" if "p_value" in row else ""), flush=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"{output['tests_run']} tests; Bonferroni threshold {threshold:.5f}")


if __name__ == "__main__":
    main()
