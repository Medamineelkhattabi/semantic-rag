## What this changes

## Why

## Checklist

- [ ] `cd backend && python -m pytest` passes
- [ ] `cd frontend && npm test && npm run build` passes
- [ ] If this touches retrieval: **both** pipelines still share the corpus,
      embedder, LLM object and answer prompt (`tests/test_fairness.py`)
- [ ] If this touches the corpus: no document holds two consecutive links of the
      Project → Component → Supplier → Contract → Risk chain
      (`test_chain_is_split_across_documents`)
- [ ] No metric is hardcoded — everything is computed from ground truth
