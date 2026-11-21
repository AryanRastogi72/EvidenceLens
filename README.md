# EvidenceLens

Search that shows its reasoning.

EvidenceLens is an Information Retrieval course project for scientific search. It searches the 171,332 papers of TREC-COVID offline: it turns a question into concepts and a Boolean query, keeps the result set between 20 and 200 papers, asks a clarifying question when the question is too broad, handles follow-up turns, labels the study type of every paper and explains why each result was ranked where it is. No generative AI is used at query time.

Repository: https://github.com/AryanRastogi72/EvidenceLens

## Novelty

The midsem project, Goldilocks, already had a result-size controller with a decision trace and session follow-ups, but it depended on a language-model API to propose and refine queries. 

EvidenceLens adds:

1. Offline query building with no language model: concept extraction by idf, a COVID synonym lexicon, a word2vec thesaurus and Rocchio feedback terms.
2. Clarifying questions made by clustering the candidate set and labelling each cluster with its most distinctive words.
3. Evidence Lens: a rule-based study-type classifier (systematic review, randomized trial, observational, case report, modelling, lab or animal, opinion, other) used as a filter and as an optional score.
4. Per-result explanations: matched words, the BM25 contribution of each, the rule behind the study type and the best sentence of the abstract.
5. BM25 ranking with a title zone, next to the tf-idf cosine ranking.
6. A larger evaluation with paired significance tests and a comparison with the BM25 result published for TREC-COVID in the BEIR paper.

These are system integration and teaching contributions built from established methods. They are not a new retrieval algorithm, and the evaluation does not show that the agent is more relevant than plain BM25 (see below).

## How search works

1. The question is lowercased, tokenized and stemmed; query stopwords are removed.
2. Each word or synonym group becomes a concept. Concepts are ordered by idf, and very common ones are held back (index elimination).
3. The conversation rules rewrite a follow-up turn into the session's concepts and show the rewritten query.
4. The controller builds a Boolean query from the three rarest concepts and counts the hits from the postings.
5. In band (20 to 200 hits): accept. Too many: add a concept, require a word in the title, or add a feedback term. Too few: add thesaurus neighbours, wildcard expansions, or drop the rarest concept.
6. If the set is still too large, the candidates are clustered and the user is asked to choose a cluster.
7. The accepted set is ranked with BM25, optionally combined with the study-type score.
8. Every result carries its explanation, and the whole trace is available behind "How did I get this?".

## Project map

| Folder or file | What it holds |
|---|---|
| `pipeline/` | tokenizer, index builder, thesaurus builder, study-type label builder |
| `engine/` | Boolean parser, controller, rankers, conversation rules, clarification, study-type classifier, explanations |
| `api/` | FastAPI app: `/api/search`, `/api/clarify`, `/api/rerank`, `/api/doc/{id}`, `/api/eval`, `/api/health` |
| `web/` | React web app |
| `evaluation/` | the six evaluation runs, significance tests, charts and saved results (`summary.md`) |
| `tests/` | unit tests |
| `resources/` | the synonym lexicon and the query stopwords |
| `scripts/` | the submission packaging script |
| `index_out/` | the prebuilt index of 171,332 documents |
| `docs/` | the report (`docs/report/EvidenceLens_Report.pdf`) and screenshots (`docs/screenshots/`) |
| `config.json` | band, BM25 parameters, thesaurus and clustering settings |

## Run on Windows

Open PowerShell in the project folder. Python 3.12 or newer and Node.js are needed.

```powershell
python -m venv .venv
& '.venv\Scripts\Activate.ps1'
python -m pip install -r requirements.txt
python -m pipeline.build_index
python -m pipeline.build_thesaurus
python -m pipeline.build_evidence
cd web
npm install
npm run build
cd ..
uvicorn api.main:app
```

Then open http://localhost:8000 (health check: http://localhost:8000/api/health). The dataset ships in the submission ZIP as `data/raw/beir/trec-covid.zip`; `build_index` reads it, or downloads TREC-COVID the first time if the file is missing. If the BEIR download server is down, the same corpus is available from the BEIR copy on HuggingFace (`BeIR/trec-covid`). `build_index` writes `index_out/`. The index files are already in the repository except `abstracts.sqlite`, so after a fresh clone only `build_index` is needed; rebuilding the thesaurus can give slightly different neighbours, because word2vec training with several workers is not bit-reproducible.

Command-line search, tests and the web development server:

```powershell
python -m engine.main "covid vaccine pregnant"
python -m unittest discover tests
cd web; npm run dev
```

To repeat the evaluation (about two minutes), then read `evaluation/summary.md`:

```powershell
python -m evaluation.run_eval
python -m evaluation.significance
python -m evaluation.make_summary
python -m evaluation.validate_results
```

## Verified evaluation

All 50 TREC-COVID topics, both the short keyword query and the question text, ranking depth 1000, nDCG with linear gain. Full tables: `evaluation/summary.md`.

| Run | Keyword P@10 | Keyword nDCG@10 | Question P@10 | Question nDCG@10 |
|---|---|---|---|---|
| R0 Goldilocks offline (AND, tf-idf) | 0.5200 | 0.4663 | 0.1360 | 0.1284 |
| R1 BM25, whole collection | 0.6660 | 0.6032 | 0.6860 | 0.6555 |
| R2 tf-idf cosine, whole collection | 0.3660 | 0.3343 | 0.3720 | 0.3392 |
| R3 BM25 + Rocchio feedback | 0.6380 | 0.5741 | 0.6680 | 0.6180 |
| R4 EvidenceLens agent | 0.6480 | 0.5897 | 0.4940 | 0.4452 |
| R5 agent + evidence weight 0.2 | 0.6460 | 0.5856 | 0.4800 | 0.4363 |

The agent returns one set rather than a ranking, so it is also scored as a set:

| Field | Run | Mean hits | Precision | Recall | F1 | In band |
|---|---|---|---|---|---|---|
| keyword | R0 | 517.5 | 0.3324 | 0.2224 | 0.2116 | 32% |
| keyword | R4 | 107.4 | 0.4656 | 0.1217 | 0.1732 | 98% |
| question | R0 | 10.1 | 0.1627 | 0.0091 | 0.0145 | 12% |
| question | R4 | 128.2 | 0.2639 | 0.0535 | 0.0776 | 96% |

The study-type classifier was checked against 120 hand-labelled documents (15 per predicted class). Estimated accuracy is 0.597 for the collection (re-weighted by sampling probability) and 0.733 on the balanced sample:

| Class | Precision | Recall (re-weighted) | F1 (re-weighted) |
|---|---|---|---|
| systematic review or meta-analysis | 0.733 | 1.000 | 0.846 |
| randomized trial | 0.800 | 0.685 | 0.738 |
| observational | 0.867 | 0.412 | 0.559 |
| case report or series | 0.733 | 0.123 | 0.211 |
| modelling or simulation | 0.933 | 0.922 | 0.927 |
| lab or animal | 0.867 | 0.671 | 0.756 |
| opinion or editorial | 0.467 | 0.142 | 0.218 |
| other | 0.467 | 0.930 | 0.621 |

What the results say, including the losses:

1. The agent keeps 98% (keyword) and 96% (question) of topics inside the 20 to 200 band, against 32% and 12% for the midsem AND baseline.
2. It is tied with BM25 on keyword queries (nDCG@10 0.5897 against 0.6032, p = 0.6737) and clearly worse on question text (0.4452 against 0.6555, p = 0.0001).
3. The band caps the set at about 200 papers, so Recall@1000 is 0.1217 and 0.0531 against 0.4161 and 0.4543 for BM25.
4. R4 and R5 run with clarification switched off, because there is no user to answer; with it on, 27 (keyword) and 7 (question) of the 50 topics would stop at a clarifying question instead of a ranked set.
5. Rocchio feedback and the evidence weight did not improve ranking. The controller revision (v2.1) was made after seeing v2.0 on all 50 topics, with no held-out topics, so its gain is a post-hoc result.
6. Our BM25 on the question text scores nDCG@10 0.6555; the BEIR paper reports 0.656 for BM25 on TREC-COVID (Table 2, version 4). Their BM25 setup differs from ours, so we do not claim to reproduce it.
7. Classifier recall is low for case reports (0.123) and opinion pieces (0.142), and the per-class numbers rest on 120 documents.

## Information Retrieval choices

1. The inverted index maps each stem to a sharded postings list of documents and term counts.
2. Boolean retrieval intersects postings lists in order of document frequency, shortest list first.
3. A trigram k-gram index expands wildcard terms, and a title-zone postings file supports title restrictions.
4. Index elimination leaves very common concepts out of the first query.
5. Okapi BM25 uses the Lucene form of idf with k1 = 1.2 and b = 0.75; the tf-idf cosine uses log tf, idf and length normalisation (ltc.ltc).
6. Rocchio pseudo-relevance feedback (positive part only) supplies tightening terms.
7. A word2vec thesaurus trained on the corpus supplies synonyms when a query is too narrow.
8. Leader-follower clustering of the top 300 candidates gives the clarifying choices, labelled by contrast scores.
9. Rule-based query rewriting handles follow-ups, exclusions, facet swaps and topic switches.
10. A rule-based classifier assigns a study type; its evidence score uses a published levels-of-evidence ordering.
11. Evaluation uses P@10, nDCG@10, MRR@10, MAP@100, recall, set precision, recall and F1, paired randomization tests and Bonferroni correction.

## Libraries

1. NLTK supplies the Porter stemmer.
2. NumPy supports the thesaurus build.
3. Gensim trains the word2vec model.
4. Requests downloads the dataset when it is not present.
5. Matplotlib writes the saved evaluation charts.
6. FastAPI and Uvicorn serve the API and the web app; HTTPX supports the API tests.
7. React, Vite, Tailwind and Motion build the web interface.

## Team

Aryan Rastogi and Raghav Garg.

## Midsem history

The midsem project, Goldilocks, is frozen in the git tag `midsem-final`.

## Data and references

TREC-COVID through BEIR (Roberts et al., Journal of Biomedical Informatics 2021; Thakur et al., 2021). The levels-of-evidence ordering behind the evidence score is from Burns, Rohrich and Chung (Plastic and Reconstructive Surgery, 2011). The full reference list is in the report.
