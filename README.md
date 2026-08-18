# RAG Intelligence Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![React 18](https://img.shields.io/badge/react-18-61dafb.svg)](https://react.dev/)
[![Semantica 0.6.5](https://img.shields.io/badge/semantica-0.6.5-8b5cf6.svg)](https://github.com/semantica-agi/semantica)
[![Tests](https://img.shields.io/badge/tests-191%20backend%20%2B%2017%20frontend-brightgreen.svg)](#tests)

An interactive lab that runs **Basic RAG** and **Semantic (graph) RAG** side by side
on the *same* corpus, the *same* embedding model and the *same* LLM — so the only
variable is the retrieval strategy.

> **Basic RAG asks:** "Which text is most similar to my question?"
> **Semantic RAG asks:** "Which entities, facts and relationships are relevant to my question?"

The semantic side is built on [**Semantica**](https://github.com/semantica-agi/semantica)
(v0.6.5) — its provider layer, `GraphBuilder`, `KnowledgeGraph` and `PathFinder`.

---

## What it demonstrates

The corpus is a synthetic enterprise dataset for a fictional aerospace company.
The supply chain forms a deliberate multi-hop chain:

```
Project Phoenix ──uses──▶ C-17 ──supplied_by──▶ Alpha Precision Systems
                              ──governed_by──▶ C-2048 ──has_risk──▶ R-17
```

**No document contains two consecutive links of that chain.** The portfolio knows
`Phoenix → C-17` but never names the supplier; the component register knows
`C-17 → Alpha` but never names the project; and so on. A test
(`test_chain_is_split_across_documents`) enforces this property, so the benchmark
cannot be accidentally trivialised by an edit to the corpus.

That is why the headline question needs a graph:

> *"Which supplier is responsible for the component used by Project Phoenix, what
> contract governs the relationship, and what risk is associated with that supplier?"*

Top-K chunk similarity retrieves text that *looks* like the question. It has no
mechanism to hop from a chunk in one document to the fact that completes it in
another.

---

## Architecture

```
                        ┌──────────────────────────────┐
                        │   12 markdown documents      │
                        │   (identical for both)       │
                        └───────────┬──────────────────┘
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
      ┌───────────────────────────┐   ┌────────────────────────────────┐
      │  BASIC RAG                │   │  SEMANTIC RAG (Semantica)      │
      │  paragraph chunking       │   │  LLM entity/relation extraction│
      │  bge-m3 embeddings        │   │  → typed ontology + provenance │
      │  FAISS IndexFlatIP        │   │  → GraphBuilder → KnowledgeGraph│
      │  top-K cosine             │   │  → entity linking (bge-m3)     │
      │                           │   │  → PathFinder multi-hop walk   │
      │                           │   │  → provenance passage expansion│
      └────────────┬──────────────┘   └───────────────┬────────────────┘
                   └──────────────┬──────────────────┘
                                  ▼
                      same LLM · same prompt shape
                                  ▼
                     answer + metrics + graph path
```

### Layout

```
backend/
  app/
    config.py                 shared settings (one source of truth)
    llm.py                    OpenAI-compatible chat client
    embeddings.py             bge-m3 client with on-disk vector cache
    service.py                engine singleton + API serialisation
    main.py                   FastAPI routes
    dataset/
      corpus.py               document loader
      benchmark.py            15 questions + ground truth
    basic_rag/pipeline.py     chunk → embed → FAISS → top-K → LLM
    semantic_rag/
      provider.py             Semantica provider registered for our gateway
      ontology.py             typed entity/relation schema
      extraction.py           LLM extraction + canonicalisation + cache
      graph.py                GraphBuilder / KnowledgeGraph / PathFinder
      pipeline.py             link → traverse → expand → LLM
    evaluation/
      metrics.py              precision, recall, F1, MRR, coverage
      runner.py               runs both pipelines over the benchmark
  data/documents/             the corpus
  tests/                      offline test suite (no network)
frontend/src/                 React + TypeScript + Tailwind + React Flow
```

---

## How Semantica is used

Everything on the semantic side goes through Semantica's real 0.6.5 API:

| Concern | Semantica API |
|---|---|
| LLM provider | `semantic_extract.OpenAIProvider`, `ProviderRegistry`, `create_provider` |
| Extraction call | `provider.generate_structured(...)` |
| Graph assembly | `kg.GraphBuilder(...).build({entities, relationships})` |
| Canonical graph type | `kg.KnowledgeGraph` |
| Multi-hop search | `kg.PathFinder.find_shortest_path(...)` |

**One deliberate extension.** `OpenAIProvider` already accepts a custom
`base_url`, but its `_init_client()` forwards only `api_key` and `base_url` to
the OpenAI SDK — it cannot attach custom auth headers. Because this gateway sits
behind Cloudflare Access, `app/semantic_rag/provider.py` subclasses the real
provider, overrides *only* client construction to add `default_headers`, and
registers the subclass through Semantica's own `ProviderRegistry`. Everything
downstream resolves it via `create_provider()` like a first-class provider.

### Chat and embeddings are configured separately

Most free chat APIs — Groq, OpenRouter, Cerebras — serve **no embedding model at
all**, so pinning embeddings to the chat endpoint makes them unusable. The
embedding backend is therefore configured independently:

```
EMBEDDING_PROVIDER=local     # fastembed, in-process, no API key
EMBEDDING_PROVIDER=api       # any OpenAI-compatible /v1/embeddings
EMBEDDING_BASE_URL=...       # optional: different provider than chat
```

With `local`, `fastembed` runs `BAAI/bge-small-en-v1.5` in-process (384-dim,
one-off ~13s model download, then cached). Fairness is unaffected — both
pipelines still share whichever embedder is configured. `.env.example` carries
copy-paste recipes for Groq, OpenRouter, Mistral and local Ollama.

### Two gateway gotchas worth knowing

**The OpenAI SDK's User-Agent gets blocked.** The Cloudflare rules in front of
some gateways reject `User-Agent: OpenAI/Python x.y.z` with a 403 while passing
the identical request under any other agent. Through the SDK that surfaces as
`PermissionDeniedError`, which the extractor would otherwise swallow into its
regex fallback — the graph still builds, so nothing looks broken, but every
relationship comes from regex rather than the LLM. `LLM_USER_AGENT` overrides it
on every client. If you point this at a different gateway and extraction
silently degrades, check `/api/status` → `extraction_methods` first: it reports
`llm`, `llm+cache` or `fallback (...)` per run.

**Bursts get 503'd.** A benchmark run issues two LLM calls per question back to
back and shared gateways answer with `429`/`503 maximum pending requests
exceeded`. All three clients (chat, embeddings, and the Semantica provider)
retry those with exponential backoff — see `LLM_MAX_RETRIES` /
`LLM_RETRY_BASE_DELAY`.

---

## Fairness

This is a comparison, not a demo rigged to flatter graphs. Specifically:

- **Same corpus.** Both call `load_corpus()`; a test asserts the sources match.
- **Same embedding model.** `bge-m3` powers Basic RAG's chunk vectors *and*
  Semantic RAG's entity linking.
- **Same LLM, same temperature, same system prompt,** and the same answer
  instruction shape.
- **Basic RAG is built properly** — paragraph-aware chunking that avoids
  mid-sentence splits, exact inner-product search over L2-normalised vectors
  (true cosine ranking, not an approximate index), and configurable top-K.
- **Same rubric.** Both are scored at document level by the identical functions
  in `evaluation/metrics.py`.
- **Easy questions are included on purpose.** Five single-hop lookups are in the
  benchmark, and Basic RAG is expected to do well on them.

Semantic RAG has a real cost, and the UI shows it: a one-off LLM extraction pass
over the corpus at build time, and higher per-query retrieval latency than a
single vector search.

---

## Metrics

All computed at run time from ground truth in `dataset/benchmark.py` — none are
hardcoded.

| Metric | Definition |
|---|---|
| **Precision** | `\|retrieved ∩ relevant\| / \|retrieved\|` over source documents |
| **Recall** | `\|retrieved ∩ relevant\| / \|relevant\|` |
| **F1** | harmonic mean of the two |
| **MRR** | `1 / rank` of the first relevant document in the ranked list |
| **Context coverage** | fraction of ground-truth facts present in the assembled LLM context |
| **Answer correctness** | fraction of ground-truth facts present in the final answer |
| **Latency** | measured per stage (retrieval / generation / total) |

Ground-truth facts are keypoints with accepted surface forms, so
`"Alpha Precision Systems S.A.S."` and `"Alpha Precision"` both count, while a
wrong supplier does not.

**Errored questions are excluded from the averages, not scored as zero.** If the
LLM gateway returns 503 mid-run, averaging that in would report an
infrastructure outage as a retrieval failure — precisely the opposite of what
the benchmark measures. The summary carries an `errored` count alongside
`questions`, and the UI shows how many runs were excluded, so a degraded run can
never be mistaken for a complete one.

---

## Measured results

One complete run, 15 questions, zero errors.
Configuration: answers `openai/gpt-oss-20b` via Groq (identical for both sides),
extraction `openai/gpt-oss-120b`, embeddings `local:BAAI/bge-small-en-v1.5`
(identical for both sides), `TOP_K=5`, `GRAPH_MAX_HOPS=5`.

| Metric | Basic RAG | Semantic RAG |
|---|---|---|
| Answer correctness | 0.672 | **0.911** |
| Recall | 0.730 | **1.000** |
| Context coverage | 0.806 | **0.978** |
| Precision | **0.534** | 0.322 |
| MRR | **0.822** | 0.772 |
| F1 | **0.558** | 0.458 |
| Mean latency | **6.4 s** | 24.9 s |

Answer correctness by hop count — the point of the exercise:

| Hops | n | Basic RAG | Semantic RAG |
|---|---|---|---|
| 1 | 5 | 1.00 | 1.00 |
| 2 | 3 | 0.83 | 1.00 |
| 3 | 2 | 0.54 | 1.00 |
| 4 | 4 | **0.21** | **0.75** |
| 5 | 1 | 0.67 | 0.67 |

Semantic RAG wins 6 questions, Basic RAG wins none, 9 are ties.

**Read this honestly.** Basic RAG is fully competitive at one hop and degrades as
the chain lengthens; that is the claim being tested, and it holds. But Basic RAG
also wins precision, MRR and F1, and is roughly 4x faster. Semantic RAG buys
perfect recall by surfacing ~10 documents where Basic surfaces 5, which drags
its precision denominator down. If your questions are single-hop lookups, the
graph is overhead you do not need. The tie at 5 hops (n=1) is a single question
and carries no weight either way.

Reproduce with `POST /api/benchmark/run`, or the **Benchmark** tab.
Numbers will move with the model and provider you configure — nothing here is
hardcoded.

---

## Quick start

### 1. Configure

```bash
cp .env.example .env
```

Edit `.env` — set `LLM_BASE_URL`, `LLM_API_KEY`, and the CF-Access headers if
your gateway needs them. Any OpenAI-compatible endpoint works (Ollama, vLLM,
OpenAI, LiteLLM, a local proxy).

### 2. Run with Docker

```bash
docker compose up --build
```

Open **http://localhost:5173**.

### 3. Or run locally

Backend:

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r backend/requirements.txt
```

```bash
cd backend && uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend && npm install && npm run dev
```

The first boot runs LLM extraction over the corpus (one call per document) and
caches the result under `backend/cache/`. Later boots load from cache and are
near-instant. The UI polls `/api/status` and shows build progress.

---

## Tests

```bash
cd backend && python -m pytest
```

186 tests, fully offline — the LLM and embedder are replaced by deterministic
fakes, so nothing touches the network. Coverage:

| Area | What it pins down |
|---|---|
| `test_dataset` | corpus loads; every ground-truth keypoint really appears in the documents it names; **no document holds two consecutive links of the chain** |
| `test_basic_rag` | chunking budgets and edge cases, cosine ranking order, retrieval shape |
| `test_semantic_graph` | id canonicalisation, cross-document entity merging, provenance, the four-hop path, reverse traversal, disconnected nodes |
| `test_extraction_validation` | ontology domain/range enforcement — shortcut edges rejected, transposed edges repaired |
| `test_path_ranking` | the explanatory chain outranks longer rambles; ranking is deterministic |
| `test_metrics` | precision/recall/F1/MRR/coverage maths; errored runs excluded from averages |
| `test_llm_client` | JSON salvage, the `reasoning`-field quirk, retry/backoff, gateway headers |
| `test_api` | every route, serialisation, and error handling |
| `test_fairness` | both sides share the corpus, embedder, LLM object and system prompt — and ground truth never leaks into a prompt |
| `test_embeddings` | cache scoping per model, normalisation, API-vs-local backend selection, retry paths |

Frontend:

```bash
cd frontend && npm test
```

17 tests over the graph layout logic — column ordering by ontology, collapsed
empty columns, explicit node dimensions, path highlighting and numbering,
reverse-direction edges, and empty/unknown-type edge cases. React Flow's own SVG
drawing is driven by `requestAnimationFrame` and a `ResizeObserver`, so it is
verified in a real browser rather than here; everything handed to it is covered.

---

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/health` | service + LLM gateway reachability |
| `GET` | `/api/status` | engine build state and stage |
| `POST` | `/api/build` | trigger a (re)build |
| `GET` | `/api/config` | the settings both pipelines share |
| `GET` | `/api/dataset` | document list |
| `GET` | `/api/dataset/{id}` | full document text |
| `GET` | `/api/benchmark/questions` | benchmark questions + ground truth |
| `POST` | `/api/compare` | run one question through both pipelines |
| `GET` | `/api/graph` | full knowledge graph |
| `POST` | `/api/benchmark/run` | score both pipelines over the benchmark |

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible base URL |
| `LLM_MODEL` | `qwen3-coder:30b` | answer model — shared by both pipelines |
| `LLM_EXTRACTION_MODEL` | `qwen3-coder:30b` | extraction model (ingest only) |
| `EMBEDDING_MODEL` | `bge-m3:latest` | embedding model — shared |
| `CF_ACCESS_CLIENT_ID` / `_SECRET` | — | Cloudflare Access service token |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `700` / `120` | Basic RAG chunking (characters) |
| `TOP_K` | `5` | chunks retrieved by Basic RAG |
| `GRAPH_MAX_HOPS` | `5` | graph expansion depth |
| `ENTITY_LINK_THRESHOLD` | `0.45` | cosine floor for embedding-based linking |

### Choosing a model

Prefer a model that returns clean JSON. Reasoning-tuned models on some gateways
put the visible answer in a non-standard `reasoning` field and leave `content`
empty; `llm.py` falls back to that field, but extraction quality is best with a
model that emits unfenced JSON directly.

---

## Notes and limitations

- Extraction quality bounds graph quality. A missed relationship is a missing
  edge, and the retrieval path will not exist. The ontology is closed-vocabulary
  and IDs are canonicalised specifically to keep this tight.
- **Entity linking treats embeddings as a fallback, not a co-equal signal.**
  When a question names an entity ("…used by Project Phoenix"), that match is
  authoritative. Adding topically-similar neighbours alongside it pulls in every
  supplier and risk in the corpus, and those spurious seeds spawn well-formed
  chains that outrank the real one. Embedding linking runs only when nothing is
  named; target-typing already steers traversal toward the types being asked for.
- **Paths are ranked for explanatory value, not length.** The score rewards
  covering the entity types asked about and crossing many distinct types, and
  penalises revisiting a type, wandering past the answer, and routing *through* a
  person. That last one matters: two contracts sharing a manager are not
  connected in the business, so `Titan → C-4501 → Anna Kowalski → R-42` would
  imply an exposure that does not exist. Ending on a person is fine — that is
  what "who owns this risk?" asks for.
- **The highlighted path is aligned to the answer.** Several chains can be
  equally well-formed (Phoenix reaches a risk via both C-17/Alpha and
  C-22/Borealis). After generation, the chain the answer actually cites is
  promoted, so the graph never contradicts the prose beside it.
- **Shortcut edges are rejected at extraction time.** Reading the Phoenix status
  report, the model reasonably emits `Project Phoenix --has_risk--> R-17`. That
  edge is not in the ontology (`has_risk` is `Contract -> Risk`), and admitting
  it would collapse the four-hop chain into one hop — the demo would keep
  working while quietly proving nothing. Every extracted relationship is
  validated against the declared domain/range: transposed endpoints are
  repaired, genuine violations are dropped and counted in
  `/api/status` → `rejected_facts`.
- `ALLOW_EXTRACTION_FALLBACK=true` lets the app boot without an LLM by using a
  regex extractor over the known document phrasing. It is a availability
  fallback for demos, not a general extractor — `/api/status` reports when a
  document was extracted this way.
- The corpus is synthetic. Every organisation, person, contract and incident in
  it is invented.

---

## License

MIT — see [LICENSE](LICENSE).

The corpus is synthetic. Every organisation, person, contract, component,
incident and risk in `backend/data/documents/` is invented for this demo and
resembles no real entity.

## Acknowledgements

The semantic pipeline is built on
[Semantica](https://github.com/semantica-agi/semantica) (MIT), using its
provider layer, `GraphBuilder`, `KnowledgeGraph` and `PathFinder`.
