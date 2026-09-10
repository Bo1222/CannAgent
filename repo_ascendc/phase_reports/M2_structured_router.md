# M2 Structured Knowledge Router

## TODO

- [x] Define `KnowledgeContext`, `KnowledgeBundle`, project-contract, and retrieval-trace schemas.
- [x] Load validated immutable knowledge builds without reading Raw Markdown.
- [x] Implement exact API identity before context, metadata, FTS, and vector-like supplemental ranking.
- [x] Prevent lexical/vector similarity from selecting an API overload.
- [x] Record selected and rejected candidates with reasons.
- [x] Preserve document routing and add `--knowledge-mode document|structured`.
- [x] Integrate structured bundles without changing planner or generator prompt logic.
- [x] Prove structured runtime routing performs no knowledge-router LLM call.

## Added structure

```text
ascendc_multi_turn/structured_knowledge/
├── router.py
└── tests/test_structured_router.py
```

Runtime structured-knowledge artifacts:

```text
.llm_state/round_NN/
├── knowledge_bundle.json
├── retrieval_trace.json
├── selected_knowledge.json
└── references.md
```

## Changes

- `RunConfig` and CLI default to structured knowledge, use the published `current.json`, and accept an explicit knowledge build ID for reproduction.
- Exact symbols select API cards and their context-matching facts; similar names are rejected explicitly.
- Metadata, lexical, and vector-like similarity are limited to failure/pattern supplements.
- Existing PLAN and generator prompt builders are unchanged.

## Validation

- Tests cover DataCopy/DataCopyPad/DataCopyExt isolation, published-build selection, trace output, and structured end-to-end mock execution without a router LLM call.
