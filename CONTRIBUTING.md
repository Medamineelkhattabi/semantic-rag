# Contributing

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r backend/requirements.txt
cd frontend && npm install
```

Copy `.env.example` to `.env` and point `LLM_BASE_URL` at any OpenAI-compatible
endpoint. The file documents working recipes for Groq, OpenRouter, Mistral and
local Ollama.

## Running

```bash
cd backend && uvicorn app.main:app --reload --port 8000
```

```bash
cd frontend && npm run dev
```

## Tests

```bash
cd backend && python -m pytest
```

```bash
cd frontend && npm test
```

Both suites run fully offline — the LLM and embedder are replaced with
deterministic fakes, so nothing touches the network and no API key is needed.

## Ground rules

- **Keep the two pipelines separate.** `basic_rag/` and `semantic_rag/` share the
  corpus, the embedder, the LLM client and the answer prompt, and nothing else.
  `tests/test_fairness.py` enforces this; if you change one side's prompt or
  model, change both or the comparison stops meaning anything.
- **Never hardcode a metric.** Everything in `evaluation/metrics.py` is computed
  from the ground truth in `dataset/benchmark.py`.
- **Keep the chain split across documents.** `test_chain_is_split_across_documents`
  fails if any single document ever contains two consecutive links of the
  Project -> Component -> Supplier -> Contract -> Risk chain. That property is
  what makes the benchmark non-trivial.
- **Extend the ontology, don't bypass it.** New relationships need an entry in
  `semantic_rag/ontology.py` with an explicit source/target signature; extraction
  rejects anything that does not match.
