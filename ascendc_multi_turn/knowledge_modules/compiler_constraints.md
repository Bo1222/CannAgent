# CannAgent compiler and memory-boundary constraints

These are stable C++/CannAgent constraints, not memories of a particular trajectory.
Exact AscendC API support still comes from the current compiler diagnostic and the
installed or compile-probed runtime facts.

## Template and dtype constraints

- A normal runtime `if` does not prevent both branches of a C++ function template
  from being instantiated. If an API is unsupported for one template dtype, isolate
  the call with a toolchain-supported compile-time dispatch or put it in a
  type-specific implementation that is never instantiated for that dtype.
- Do not generalize one observed API/dtype failure to other APIs or environments.
  The unsupported pair must be named by direct compiler evidence or a matching
  installed/Verified fact.
- When repairing an overload, preserve already compiled dtype paths. Change the
  smallest call site or specialization that owns the failure.

## Host and NPU memory boundary

- A `GM_ADDR` kernel argument is a device/global-memory address. A pointer to a Host
  stack array, `std::vector` storage, or other CPU-owned descriptor is not a valid
  `GM_ADDR`, even when a C++ cast makes the wrapper compile.
- Shape, stride, permutation, and tiling descriptors consumed by the kernel must have
  device-visible storage with a lifetime covering asynchronous kernel execution.
- In the CannAgent PyTorch binding, build small descriptors on CPU and transfer them
  to the NPU through the locally verified tensor/device path before passing their
  device data pointers to the launch wrapper.
- Preserve ownership and lifetime across the launch. Do not pass a temporary Host
  object's address and do not free device descriptor storage before stream work can
  consume it.
- Preserve `__gm__` on kernel/helper parameters that consume GM descriptors. An
  ordinary pointer and a global-memory-qualified pointer are different ABI domains.
- The pybind declaration, `*_do` definition, and kernel entry are one signature
  contract. Argument count/order/layout changes must be applied consistently.

## Compile ownership constraints

- A diagnostic in `pybind11.cpp`, a Host include, an ATen container expression, or a
  wrapper declaration is a Host/boundary compile failure unless the message directly
  identifies an AscendC kernel API.
- A diagnostic in a kernel source that identifies an AscendC overload, template,
  address-space qualifier, dtype restriction, or owner is a Kernel API compile failure.
- Generic words such as `COMPILER`, `Traceback`, `Tensor`, `PYBIND11_MODULE`, and local
  project function names are not installed AscendC API symbols.

## Transactional repair rule

- If a runtime or correctness hypothesis introduces a compile error, repair that
  immediate compiler error on the same candidate branch. Preserve the parent
  hypothesis and all unrelated compiled regions.
- A failed branch is episodic trajectory state. It must not be promoted into this
  stable knowledge module unless its constraint is independently supported by compiler,
  installed-header, project-contract, or compile-probe evidence.
