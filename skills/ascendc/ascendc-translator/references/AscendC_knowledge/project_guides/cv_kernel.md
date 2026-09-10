# Direct AscendC Cube/Vector Kernel Pattern

Authority: PROJECT_CONTRACT.

For mixed AIC/AIV kernels, make the core-role branch explicit and derive the number of cube
and vector workers from tiling. Define workspace ownership and producer/consumer ordering
before implementing stages. Do not let AIC and AIV write overlapping output or reuse a
workspace slot without a proven synchronization edge.
