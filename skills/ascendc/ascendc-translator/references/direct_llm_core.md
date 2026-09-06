# AscendC Direct-LLM Core Rules

This is the compact, invariant rule set for direct multi-turn generation. Raw
guides and API pages are evidence sources loaded only when relevant.

## Source of truth

1. Compiler diagnostics and declarations from the installed CANN public headers
   have priority over bundled documentation.
2. Version-fallback documentation supplies semantics and examples, not guaranteed
   field names or overload signatures.
3. When a runtime declaration conflicts with a document, follow the declaration
   and mention the conflict in the implementation analysis.

## Kernel structure

- Keep core computation in AscendC. Python and pybind code may validate inputs,
  allocate tensors/workspace, prepare tiling, and launch the kernel only.
- `TQue::EnQue` receives the allocated `LocalTensor`; `TPipe` initializes buffers
  and does not own queue enqueue/dequeue operations.
- Pair `AllocTensor -> data movement -> EnQue` and
  `DeQue -> computation -> FreeTensor` consistently.
- `TBuf` is for calculation scratch tensors; `TQue` is for staged input/output.
- Do not copy a struct from `__gm__` to local memory with ordinary C++ assignment.
  Use a supported registered tiling mechanism or read supported scalar fields.
- Treat convenience helpers such as `CopyTiling` as type- and component-specific;
  do not assume an internal Matmul helper supports an arbitrary tiling struct.

## API use

- Match every function call against the exact runtime overload, including argument
  count, template constraints, parameter-structure family, and units.
- Do not mix `DataCopyParams` with `DataCopyPadExtParams`. Extended padding uses
  `DataCopyExtParams + DataCopyPadExtParams<T>`; the non-extended overload uses
  `DataCopyParams + DataCopyPadParams`.
- Prefer aggregate constructors for parameter structs when documentation and
  runtime field spellings may differ.
- `DataCopyParams.blockLen` and `DataCopyExtParams.blockLen` can use different
  units. Verify the selected overload before computing copy lengths.
- Recheck every call site after changing a helper signature. Analysis claims are
  not evidence; the final emitted source must satisfy the compiler.

## Repair discipline

- Resolve every distinct compiler diagnostic, including earlier diagnostics that
  may be absent from a terminal tail.
- Avoid replacing an unknown API with an invented member or overload. Consult the
  runtime declaration first.
- Preserve working code and return the smallest complete file delta.
