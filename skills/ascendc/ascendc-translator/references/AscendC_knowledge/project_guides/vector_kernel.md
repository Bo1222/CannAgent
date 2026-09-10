# Direct AscendC Vector Kernel Pattern

Authority: PROJECT_CONTRACT.

Bind GM tensors in `Init`, initialize `TPipe`, `TQue`, and reusable `TBuf` storage, then
organize `Process` as bounded CopyIn, Compute, and CopyOut stages. Queue ownership stays on
`TQue`: allocate/enqueue input, dequeue/compute/free input, allocate/enqueue output, and
dequeue/copy/free output. Derive per-core and tail lengths from tiling; verify the exact
DataCopy overload and units before handling unaligned tails.
