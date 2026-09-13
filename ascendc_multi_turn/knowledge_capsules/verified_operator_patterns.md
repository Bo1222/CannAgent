# CannAgent operator semantic invariants

These invariants constrain reasoning. They are not complete implementation templates.
API spellings and overloads must come from matching installed or compile-probed runtime
facts, not from this document alone.

## GELU semantic invariants

- Read the reference model to determine whether the requested definition is exact or
  tanh-approximate; never silently substitute one for the other.
- For tanh approximation preserve the mathematical grouping and constants from the
  reference. Do not change the cubic term, scale, or final `0.5*x*(1+...)` ordering
  without an equivalence argument.
- State the input/output dtype and the arithmetic/accumulation dtype. Cast at explicit
  boundaries and retain enough temporary LocalTensor storage for every live value.
- Ground every Tanh, Mul, Muls, Add and Cast call independently against runtime facts;
  scalar type, argument order, count/mask form, and temporary workspace are part of the
  contract.
- Partition the logical element count exactly once. Mask or safely stage the last tile;
  padded elements must not affect visible output.

## LayerNorm semantic invariants

- Normalize over exactly the dimensions used by the reference model. Define the row
  count and normalized width from that contract before choosing blocks or tiles.
- Compute `mean = sum(x)/N` and the reference variance definition over the same N.
  Apply epsilon at the same mathematical location as the reference before reciprocal
  square root.
- State the reduction and accumulation dtype. A lower-precision input does not imply a
  lower-precision accumulator.
- Map gamma and beta by normalized coordinate, not by global linear offset. Their
  broadcasting must repeat identically for every outer row.
- ReduceSum workspace/shared temporary storage and sqrt/rsqrt paths must be grounded in
  the installed/probed API facts. Host `sqrtf` is not an AICore implementation path.
- Tail handling must prevent padded values from contributing to mean or variance.

## Permute semantic invariants

For input dimensions `in_dim[0..R-1]` and permutation `p[0..R-1]`:

- `out_dim[o] = in_dim[p[o]]`.
- Build inverse permutation so `inverse[p[o]] = o`.
- Compute contiguous strides from the last dimension toward the first for both input
  and output shapes.
- Convert output linear index `q` to output coordinates using output strides.
- Map coordinates with `input_coord[p[o]] = output_coord[o]`.
- Compute source index as `sum(input_coord[i] * input_stride[i])`.
- Use an integer width that cannot overflow the largest tested element count/stride.
- Do not assume the source indices of an output tile are contiguous. Bulk DataCopy is
  valid only for a branch whose source stride is proven to be one; otherwise use a
  correct gather/scalar/indexed strategy supported by the target.
- Define the supported rank bound and preserve arbitrary-rank behavior within that
  bound. Do not hard-code one test permutation.
- Process each output element exactly once and mask/tail the final transfer without
  reading or writing outside the logical tensor.

Minimal reasoning skeleton: derive dimensions and strides, prove the coordinate map on
a small hand-worked shape, then implement that same formula. A compiled kernel that
executes but produces a broad mismatch remains a kernel-design failure, not automatic
evidence of a precision problem.
