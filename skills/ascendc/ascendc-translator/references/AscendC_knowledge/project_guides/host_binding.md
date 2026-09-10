# Direct AscendC Host Binding Contract

Authority: PROJECT_CONTRACT. These rules describe this repository's build and validation ABI.

## Python extension binding

Use a literal `PYBIND11_MODULE(module_name, m)` declaration. `model_new_ascendc.py`
must use a module-level literal `import module_name`, and `Model.forward()` must call
`module_name.exported_function(...)` directly or through a statically traceable helper.
Do not use dynamic `importlib` loaders or rename the extension through an opaque factory.

## Host wrapper and launch

`kernel/pybind11.cpp` declares and calls an `extern "C"` function ending in `_do`.
The matching definition is colocated with the `__global__ __aicore__` kernel and launches
it with `kernel<<<blockDim, nullptr, stream>>>(...)`. Template kernel launches are valid.
Do not use `ACLRT_LAUNCH_KERNEL` or include `acl/acl_rt_launch.h`.

## Host responsibilities

The binding validates metadata, allocates output/workspace, prepares tiling values, obtains
the current NPU stream, and invokes the wrapper. Core tensor computation remains in AscendC.
Only include headers present in the project or active CANN installation; do not guess local
`platform_ascendc.h` paths.
