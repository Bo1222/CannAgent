# M4 API Constraint Validator

## TODO

- [x] Add `ResolvedApiCall` and a shared `ApiCallResolver`.
- [x] Parse balanced AscendC calls, arguments, visible types, parameter structures, and overload facts.
- [x] Resolve only exact API/context facts and retain source fact IDs.
- [x] Add generic unit, alignment, and project-contract checks.
- [x] Avoid API-name-specific validation branches.
- [x] Run API constraint validation before static validation and compilation.
- [x] Persist `resolved_api_calls.json` and API constraint validation logs.
- [x] Run M4 and regression tests.

## Added structure

```text
ascendc_multi_turn/structured_knowledge/
├── api_call_resolver.py
├── api_constraint_validator.py
└── tests/test_m4.py

.llm_state/round_NN/
├── resolved_api_calls.json
└── api_constraint_validation.log
```

## Changes

- The resolver binds exact API calls to parameter-structure/overload facts and returns their provenance IDs.
- The validator interprets generic API constraints rather than embedding DataCopyPad rules in Python.
- Local evaluation invokes API constraint validation only in structured knowledge mode and before any compiler process.

## Validation

- Tests prove DataCopyPad is not contaminated by a DataCopy fact, resolve its overload and `blockLen` unit, reject an element-count expression where bytes are required, and enforce generic stream/launch contracts.
