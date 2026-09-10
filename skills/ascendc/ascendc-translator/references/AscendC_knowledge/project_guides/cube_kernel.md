# Direct AscendC Cube Kernel Pattern

Authority: PROJECT_CONTRACT.

Bind GM and tiling fields in `Init`, map A1/A2/B1/B2/CO1 positions explicitly, and size every
queue from the selected tile. Keep LoadData, Mmad, and Fixpipe parameter semantics tied to the
resolved CANN overload and SoC. `Process` must bound work by the current core's assigned tiles.
