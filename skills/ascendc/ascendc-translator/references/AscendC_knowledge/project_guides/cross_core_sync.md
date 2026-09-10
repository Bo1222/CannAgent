# Direct AscendC Cross-Core Synchronization Pattern

Authority: PROJECT_CONTRACT.

Choose bulk synchronization or a bounded workspace queue explicitly. For every flag, record
producer, consumer, slot, set/wait order, reset policy, and reuse boundary. Diagnose hangs by
checking worker counts, slot indexing, missing producers, waits on stale generations, and
workspace overlap before changing API parameters.
