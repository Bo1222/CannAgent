# CannAgent broadcast semantic contract

This knowledge module is a semantic constraint, not a copyable kernel. API spellings must be
grounded in installed headers or a matching compile probe.

## Shape, stride, and offset invariants

- Right-align input shapes to the output rank. A dimension is compatible only when
  it equals the output dimension or equals one.
- Compute contiguous storage strides first, then set the logical stride of every
  broadcast dimension to zero. Collapse dimensions only when the resulting affine
  coordinate-to-offset mapping is preserved for every input.
- For output coordinate `o`, an input coordinate is `0` on a broadcast dimension
  and `o[d]` otherwise. The input offset is the sum of those coordinates times the
  input storage strides.
- A zero logical stride means reuse one stored value. It does not mean that a whole
  output tile may be read contiguously from that input.
- For `[128,128] + [128,1]`, each output row reads exactly one `y[row,0]` value and
  expands it over 128 columns. A contiguous read of 128 `y` elements for one row is
  out of range and implements the wrong map.

## Data movement invariants

- Prove a source span is contiguous before using bulk `DataCopy`. Its last addressed
  byte must stay inside the actual input allocation; output tile length alone never
  proves an input read length.
- OneDim broadcast is valid only when the repeated axis and outer offset match its
  assumptions. UB Broadcast and NDDMA are alternative implementations after the
  semantic map is proven, not substitutes for that proof.
- Preserve the same input-offset formula for full tiles and tails. Padding and aligned
  transfers must not read beyond the logical source span or make padded values visible.

## Forbidden broadcast patterns

- Do not turn `stride == 0` into `DataCopy(local, input + outer, output_tile_len)`.
- Do not infer that both operands are contiguous merely because the output is
  contiguous.
- Do not route a broad, shape-dependent mismatch to precision debugging unless dtype,
  cast, tolerance, overflow, or rounding evidence is explicit.
