"""The TREC-COVID topics and judgments, and the two topic fields every evaluation script uses."""

import json
import os

QUERIES_PATH = os.path.join("data", "raw", "beir", "trec-covid", "queries.jsonl")
QRELS_PATH = os.path.join("data", "cache", "qrels.json")
FIELDS = {"metadata.query": lambda topic: topic["metadata"]["query"], "text": lambda topic: topic["text"]}


def load_topics():
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_qrels():
    with open(QRELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
