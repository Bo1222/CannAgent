# M1 Offline Structured Knowledge Build

## TODO

- [x] Add immutable staging/validate/publish knowledge construction.
- [x] Add API, failure, and pattern card schemas.
- [x] Require evidence, applicability, source hash, and valid source sections.
- [x] Report conflicts only for identical API/context keys.
- [x] Build exact symbol indexes and knowledge-build manifests.
- [x] Add the offline knowledge build CLI.
- [x] Verify DataCopy, DataCopyPad, TPipe, and TQue availability.
- [x] Run M1 and repository regressions.

## Added structure

```text
ascendc_multi_turn/structured_knowledge/
├── build.py
├── knowledge_build.py
├── validate.py
└── tests/test_knowledge_build.py

ascendc_multi_turn/knowledge/build.py  # stable public CLI entry point

knowledge_store/cann/<version>/
├── current.json
└── builds/<knowledge_build_id>/
    ├── raw/
    ├── normalized/
    ├── facts/
    ├── cards/
    ├── indexes/
    ├── build_manifest.json
    └── validation_report.json
```

## Changes

- Knowledge build IDs derive from canonical source hashes and compiler/schema versions.
- Repeated identical builds reuse the immutable published build and update `current.json`.
- Context keys prevent same-named parameters in different tables/overloads from being merged.
- Symbol discovery tolerates official Markdown exports that squash declarations such as `TPipe pipe` into `TPipepipe` without changing the Raw evidence.
- The compiler is exposed through `python -m ascendc_multi_turn.knowledge.build`; `structured_knowledge.build` remains the implementation module.

## Validation

- Tests cover immutable rebuilds, published-build loading, provenance validation, required API symbols, real-vs-context conflict behavior, and public CLI discovery.
