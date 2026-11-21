"""
size_matched.py
A diagnostic, not part of the main runs: "why not just show BM25's top N?"

For every topic and field, the agent (R4) returns a set of k documents. This script takes BM25's top k documents over
the whole collection (the R1 setting) and scores that set exactly as R4's set is scored (set precision, recall, F1).
Then it runs the paired randomization tests of significance.py on set F1 and set precision.

Usage: python -m evaluation.size_matched   (writes evaluation/size_matched.json, changes no other file)
"""

import json
import os
import random
import sys
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.controller import load_config
from engine.ranker import rank_bm25
from engine.search import SearchIndex
from evaluation.metrics import compute_set_metrics
from evaluation.significance import ALPHA, SEED, compare
from evaluation.topics import FIELDS, load_qrels, load_topics
from pipeline.tokenizer import query_terms

RESULTS_PATH = os.path.join("evaluation", "results.json")
OUTPUT_PATH = os.path.join("evaluation", "size_matched.json")
TESTED_METRICS = ["set_f1", "set_precision"]
SET_COLUMNS = ["hits", "set_precision", "set_recall", "set_f1"]


def bm25_top_k(text, k, index, config):
    if k == 0:
        return set()
    ranked = rank_bm25(query_terms(text), index, top_k=k, k1=config["bm25_k1"], b=config["bm25_b"], title_boost=0.0)
    return {index.get_ext_id(doc_id) for _, doc_id in ranked}


def size_matched_field(field_getter, topics, agent_topics, qrels, index, config):
    matched = {}
    for topic in topics:
        topic_id = topic["_id"]
        relevant = {ext_id for ext_id, grade in qrels[topic_id].items() if grade > 0}
        k = int(agent_topics[topic_id]["hits"])
        scores = compute_set_metrics(bm25_top_k(field_getter(topic), k, index, config), relevant)
        scores["hits"] = k
        matched[topic_id] = scores
    return matched


def means(per_topic):
    return {column: mean(float(record[column]) for record in per_topic.values()) for column in SET_COLUMNS}


def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        agent = json.load(f)["runs"]["R4"]["fields"]
    config, index = load_config(), SearchIndex()
    topics, qrels = load_topics(), load_qrels()
    rng = random.Random(SEED)
    output = {
        "note": "diagnostic only: BM25 top k over the whole collection with k = the size of R4's accepted set; no other result file changes",
        "fields": {}, "tests": [],
    }
    for field, getter in FIELDS.items():
        agent_topics = agent[field]["per_topic"]
        matched = size_matched_field(getter, topics, agent_topics, qrels, index, config)
        output["fields"][field] = {"agent_R4": means(agent_topics), "bm25_top_k": means(matched)}
        pair = {"runs": {"R4": {"fields": {field: {"per_topic": agent_topics}}}, "BM25_TOP_K": {"fields": {field: {"per_topic": matched}}}}}
        for metric in TESTED_METRICS:
            output["tests"].append(compare(pair, "R4", "BM25_TOP_K", field, metric, rng))
    output["tests_run"] = len(output["tests"])
    output["bonferroni_threshold"] = ALPHA / output["tests_run"]
    for test in output["tests"]:
        test["significant_after_bonferroni"] = test["p_value"] < output["bonferroni_threshold"]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    for field, values in output["fields"].items():
        print(field, {name: {k: round(v, 4) for k, v in row.items()} for name, row in values.items()})
    for test in output["tests"]:
        print(f"{test['field']:15} {test['metric']:14} R4 vs BM25 top-k: diff {test['mean_difference']:+.4f}  p={test['p_value']:.4f}  "
              f"R4 wins {test['topics_a_wins']}, BM25 wins {test['topics_b_wins']}, tied {test['topics_tied']}")
    print(f"{output['tests_run']} tests; Bonferroni threshold {output['bonferroni_threshold']:.4f}")


if __name__ == "__main__":
    main()
