# CannAgent Host C++ contract

## Container and type facts

- Use the exact type declared by the installed ATen/torch_npu headers and by the current
  expression. Do not assume an arbitrary vector-like shape/container exposes `.vec()`.
- Convert or copy a shape only through operations supported by its actual return type.
  Compiler diagnostics at the failing Host expression are authoritative.
- A Host C++ type error is owned by Host integration. Keep kernel arithmetic, tiling,
  and data movement unchanged while repairing it.

## Build and include facts

- Include paths and platform APIs must exist in the current CannAgent build include
  directories or be confirmed by an installed-header probe.
- Do not infer a platform include from a CANNBot sample, another CANN release, CUDA, or
  a sibling project.
- Preserve the pybind module name, wrapper declaration, and already compiled kernel
  regions when the error is confined to Host container or include handling.

## Forbidden Host patterns

- Do not add `.vec()` solely because another ATen shape API returned a vector in a
  different version.
- Do not repair Host compilation by casting CPU-owned descriptor storage to `GM_ADDR`.
- Do not inject unrelated AscendC arithmetic tutorials for a Host-only diagnostic.
