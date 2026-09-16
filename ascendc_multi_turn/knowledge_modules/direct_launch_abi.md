# CannAgent direct-launch ABI contract

This knowledge module combines documented direct-invoke concepts with CannAgent's local build
contract. Installed headers, current project source, and successful local probes have
higher authority than CANNBot examples.

## Three-way signature invariant

- The declaration in `kernel/pybind11.cpp`, the `extern "C" *_do` definition, and the
  `__aicore__` kernel entry must agree on argument count, order, width, constness where
  applicable, and the meaning of every descriptor argument.
- The wrapper definition lives beside the kernel and launches with the project form
  `kernel<<<blockDim, nullptr, stream>>>(...)`. Do not substitute registry launch or an
  unrelated CANNBot project scaffold.
- Changing a descriptor layout requires changing all three participants together.
  A declaration-only cast is not an ABI repair.

## Host, GM, and descriptor boundary

- `GM_ADDR` and `__gm__ T*` denote NPU global-memory addresses. A Host stack address,
  `std::vector::data()`, or ordinary `const void*` is not converted into device-visible
  storage by a C++ cast.
- Kernel helpers that consume global-memory descriptors must preserve the `__gm__`
  address-space qualifier in their parameters. Do not erase it into an ordinary Host
  pointer type.
- Shape, stride, permutation, and tiling descriptors read by the kernel must reside in
  NPU-visible storage and remain alive until asynchronous execution has consumed them.
- Use the locally verified tensor allocation/copy/data-pointer chain. Never apply an
  unverified `const_cast`, C-style cast, or `reinterpret_cast` from `const void*` to
  `GM_ADDR` merely to silence the compiler.

## Stream and authority rule

- Acquire and unwrap the stream only through the exact installed torch_npu declaration
  or a matching local probe. CANNBot examples containing `stream(true)`,
  `stream(false)`, or `stream()` are not CannAgent facts and must not override the
  project contract.
- Keep descriptor ownership alive across the launch on that stream. If lifetime is not
  proven, synchronization or ownership must be fixed rather than hidden behind a cast.
