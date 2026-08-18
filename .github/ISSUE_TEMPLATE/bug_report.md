---
name: Bug report
about: Something behaves differently than documented
labels: bug
---

**What happened**

**What you expected**

**To reproduce**
1.
2.

**Environment**
- Running via: Docker / local
- `LLM_BASE_URL` provider (Groq, OpenRouter, Ollama, …):
- `LLM_MODEL`:
- `EMBEDDING_PROVIDER`: api / local

**Useful output**

`GET /api/status` — in particular `extraction_methods` and `rejected_facts`:

```json

```

> If extraction silently fell back to regex, `extraction_methods` will show
> `fallback (...)`. That is usually a gateway rejecting the request rather than
> a bug in the pipeline — see the gateway notes in the README.
