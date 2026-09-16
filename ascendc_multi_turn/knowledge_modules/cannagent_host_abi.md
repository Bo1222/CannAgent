# CannAgent Host ABI knowledge module

This knowledge module adapts direct-invoke knowledge to CannAgent. It is reference knowledge,
not a workflow. Exact signatures are usable as authoritative facts only when the
runtime fact registry reports a matching environment fingerprint and successful
compile probe.

## Positive Host ABI facts

- Validate NPU tensors with the torch_npu device utilities available in the installed
  headers. The current installed header declares `torch_npu::utils::is_npu(const
  at::Tensor&)`; require the matching `runtime:is_npu` fact before treating this exact
  spelling as Verified.
- Obtain the current stream through the installed torch_npu NPU stream interface. The
  current installed header declares `c10_npu::getCurrentNPUStream(...)` and stream-handle
  accessors; use only the overload and accessor shown by matching runtime/probe facts.
- Include the exact torch_npu headers named by the runtime facts. Do not infer include
  paths from a different torch_npu release.
- `kernel/pybind11.cpp` performs validation, output/workspace allocation and calls an
  `extern "C" *_do` wrapper. The matching wrapper definition lives beside the
  `__aicore__` kernel and launches it with the project-supported triple-chevron form.
- Keep the `PYBIND11_MODULE` literal consistent with the module imported by
  `model_new_ascendc.py`.

Project provenance: `utils/build_ascendc.py`, `ascendc_multi_turn/prompts.py`, and
working archived CannAgent pybind sources. Installed-header provenance is emitted by
`runtime_knowledge.py`; successful probe provenance is emitted by `knowledge_probe.py`.

## Negative Host ABI facts

- Do not include CUDA headers or use `at::cuda`, `c10::cuda`, `CUDAContext`, or
  `CUDAStream`.
- Do not validate an NPU tensor with `.is_cuda()`.
- Do not call `getCurrentCUDAStream()`.
- Do not call `aclrtGetCurrentStream()` in this project; it was a failed generated
  pattern and is not an allowed source of the current torch_npu stream.
- Do not copy CANNBot CMake, registry, run-script, or add-custom scaffolding into a
  CannAgent candidate.

## Host repair checklist

1. Identify whether the failure is include, tensor-device check, stream acquisition,
   stream-handle conversion, wrapper declaration/definition, link, or module import.
2. Use the highest-confidence matching runtime fact for only that layer.
3. Remove every CUDA-only include/type/call.
4. Keep kernel math and tiling unchanged while repairing Host ABI.
5. Keep the pybind declaration, `extern "C" *_do` definition, and kernel entry argument
   order/layout consistent; do not patch one side in isolation.
6. Treat Host shape/container expressions according to their exact installed type. Do
   not assume every vector-like value provides `.vec()`.
7. A descriptor read by the kernel must use NPU-visible storage whose ownership covers
   asynchronous launch. Never cast a Host descriptor pointer to `GM_ADDR`.
