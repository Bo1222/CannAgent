# CannAgent kernel memory contract

## Range invariants

- Every `GlobalTensor` logical length, GM base offset, `DataCopy`/`DataCopyPad`
  `blockLen`, block count, stride, and tail span must fit the actual allocation passed
  at launch.
- Derive the maximum byte address touched for each input, output, and descriptor before
  changing synchronization or API overloads. Include alignment rounding and the final
  partial block in the derivation.
- A descriptor address must be NPU-visible and its lifetime must cover asynchronous
  execution before its fields or derived offsets can be trusted.

## 507001 and 507035 diagnosis order

1. Confirm the wrapper/kernel signature and that each address names the intended NPU
   allocation.
2. Confirm descriptor storage, layout, byte size, and lifetime.
3. Compute GM offsets and source/destination spans for the exact failing shape,
   including broadcast stride-zero inputs and tails.
4. Check block length, block count, source/destination stride, alignment, and UB capacity.
5. Only after ranges are proven, investigate pipeline synchronization or a device/API
   defect.

## Separation from compile repair

- A cast or signature change that merely compiles does not validate a runtime address.
- Keep runtime range hypotheses separate from overload/template repairs. When a runtime
  idea creates a compile error, repair the compiler error without discarding the range
  invariant or changing unrelated kernel math.
- Never increase a `GlobalTensor` length or copy count beyond the allocation to suppress
  bounds symptoms.
