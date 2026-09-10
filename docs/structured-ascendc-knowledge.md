# AscendC Structured Knowledge Build and Runtime Use

## Two explicit runtime modes

`--knowledge-mode structured` is the default. It reads validated structured knowledge and
does not ask an LLM to choose raw Markdown documents. `--knowledge-mode document` is the
diagnostic fallback: it selects API Markdown and inserts the selected text into the model
prompt.

## Install or update structured knowledge

Run this once after cloning the repository and again only when the source documents, project
contracts, schema, compiler, or target CANN documentation version changes:

```bash
python -m ascendc_multi_turn.knowledge.build --version 8.5.0
```

The default source is
`skills/ascendc/ascendc-translator/references/AscendC_knowledge`, and the default destination
is the repository-local `knowledge_store`. `--source` and `--output` remain available for a
different official-document version or installation location.

The command performs these deterministic stages:

1. Discover every source Markdown file and calculate its SHA-256 content hash.
2. Parse heading hierarchy, paragraphs, tables, code blocks, document ID, source path, source
   URL, and evidence locations into `NormalizedDocument` records. This parsing does not use an
   LLM.
3. Extract parameter-table statements into `AtomicFact` records. Each fact carries subject,
   predicate, value, API/overload/parameter context, authority, schema version, and provenance.
4. Reject facts without evidence, applicability, source hash, or a valid source section. Report
   mutually exclusive facts only when their API and applicability context are identical.
5. Validate `project_knowledge.json` evidence against the project guides, then compile
   `ProjectContract`, `FailureCard`, and `PatternCard` records with `PROJECT_CONTRACT`
   authority.
6. Build exact API cards and the exact symbol index. Project-guide words are excluded from the
   API symbol index.
7. Write `validation_report.json`. An invalid build is not published.
8. Derive `knowledge_build_id` from source hashes, CANN version, schema version, and compiler
   version. Publish the immutable build and update `current.json` only after validation passes.

The installed result is:

```text
knowledge_store/
└── cann/
    └── <CANN-knowledge-version>/
        ├── current.json
        └── builds/
            └── <knowledge_build_id>/
                ├── raw/
                ├── normalized/
                ├── facts/atomic_facts.json
                ├── cards/api_cards.json
                ├── cards/failure_cards.json
                ├── cards/pattern_cards.json
                ├── cards/project_contracts.json
                ├── indexes/symbols.json
                ├── build_manifest.json
                └── validation_report.json
```

`raw` is retained for audit and evidence lookup. Kernel generation reads the other compiled
records; it does not parse the full `raw` tree. Rebuilding unchanged input produces the same ID.
Changing input produces a new immutable directory, and `current.json` points normal runs to the
new validated build. `--knowledge-build-id` is only needed to reproduce an older experiment.

## Effect during Kernel generation

For every candidate, the runner creates `KnowledgeContext` from the operator, phase, runtime
CANN version, knowledge version, SoC, current AscendC symbols, active plan, and structured
failure. `StructuredKnowledgeRouter` then:

1. resolves exact API names through `indexes/symbols.json`;
2. selects only facts whose API applicability matches those names;
3. selects failure cards from structured failure signals;
4. ranks project patterns using the current operator, phase, symbols, failure, and plan;
5. always includes repository project contracts and their evidence provenance;
6. writes `knowledge_bundle.json` and `retrieval_trace.json` for audit;
7. renders only that bounded bundle into the Planner/Generator/Diagnose context.

Before compilation, `ApiConstraintValidator` resolves actual AscendC calls against the same facts
and rejects detectable parameter-unit, alignment, and project-contract violations. This means
the LLM reasons over selected verified constraints while deterministic code enforces constraints
that can be checked statically.

The current implementation provides exact symbol lookup, applicability filtering, failure-card
matching, and token-based pattern ranking. It does not yet provide a complete BM25 engine or an
embedding vector database; similarity is never allowed to choose the final API identity.
