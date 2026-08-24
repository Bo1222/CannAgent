
---

# 13. 最新可优化方向、动机与方案排序

> 本节是在完整 literature landscape、Ascend 工程可行性，以及最新“Host/Kernel 双边探测成本”讨论后的最终排序。

## 13.1 第一优先级：Conditional Reasoning + Plan Amortization

### 动机

很多长序列 agent 把“重新分析方向”和“生成下一版代码”绑定成同一个高成本调用。

研究真正的问题：

> **什么时候值得重新思考？**

而不是：

> 每一步都应该更深地思考什么？

### 方案

```text
Planner
→ persistent plan
→ cheap executor edits × N
→ hardware evidence
→ continue / repair / replan
```

### 为什么比 Host-vs-Kernel Routing 更直接

它不需要额外双边 probes，直接优化：

\[
\#PlannerCalls
\]

和：

\[
T_{\text{planning}}.
\]

### 评价

- 新颖性：**中高**
- 实现难度：**中**
- Token relevance：**很高**
- Ascend 适配性：**高**

---

## 13.2 第二优先级：Patch / Delta Executor

### 动机

即使 Planner 调用减少，如果 Executor 每轮仍重新输出几百行 AscendC，output token 和 structural drift 仍然很高。

### 方案

只输出：

```text
unified diff
```

或结构化 patch/action。

### 评价

- 新颖性：**中**
- 实现难度：**低–中**
- 与 Conditional Reasoning 的互补性：**很高**

---

## 13.3 第三优先级：Selective Hardware Evidence / Profiler Filtering

### 动机

CudaForge/CUDAMaster 已经说明 full profiler 并不总是值得。

### 方案

默认只使用：

```text
runtime
compile/correctness
```

只有 plan uncertainty 高时：

```text
LIGHT_PROFILE
→ optional FULL_PROFILE
```

### 评价

- 新颖性：**中高**
- 实现难度：**中**
- 第一版：**可不做**

---

## 13.4 第四优先级：Headroom-Aware Stop / Budgeting

### 动机

μCUTLASS+SOL 已直接证明 diminishing-return / headroom budgeting 可以节省 19–43% token，同时保持至少 95% geomean speedup。

### 最新判断

这个方向本身已有强 precedent，所以不建议单独做“early stop”论文。

但它是 Conditional Reasoning 非常重要的 baseline：

> “为什么你的 Planner trigger 比 SOL / patience stop 更有价值？”

### 评价

- 新颖性：**中低（单独）**
- Baseline 价值：**很高**

---

## 13.5 第五优先级：Structured Strategy Controller / Bandit

### 动机

KernelBand 已经证明结构化 action policy 可以减少 blind exploration。

### 方案

把优化方向变成有限动作：

```text
TILING
VECTORIZE
PIPELINE
MEMORY
PROFILE
STOP
```

controller 可以是 bandit/小模型，而不是大 LLM。

### 评价

- 新颖性：**中**
- Token 潜力：**高**
- 与 Planning 的关系：可作为非 LLM Planner baseline。

---

## 13.6 第六优先级：Host / Kernel Decomposition

### 最新定位

**不再作为 token-saving 主方向。**

它适合回答 Ascend-specific 的问题：

> Host tiling 与 Device Kernel 是否存在不同的收益阶段和协同？

### 推荐实验

只在少量 checkpoint 离线做：

```text
fix kernel → small host budget
fix host   → small kernel budget
```

得到：

\[
G_H(s),G_K(s)
\]

用于：

- analysis；
- oracle；
- future controller training。

### 不推荐

在线每轮先把两边都 probe 后再决策。

### 评价

- Ascend-specific：**很高**
- 作为主 token-saving novelty：**中低**
- 作为 case study：**高**

---

## 13.7 第七优先级：Planner / Executor Model Routing

例如：

```text
V4-Pro Planner
+
V4-Flash Executor
```

但应放在证明 Plan/Execute separation 有价值之后。

---

## 13.8 第八优先级：Memory / Skill Distillation

KernelBlaster、KernelSkill、AscendOptimizer 已经证明 memory 有价值。

更值得做：

```text
trajectory / plan
→ compact strategy prior
```

而不是再建一个大 RAG。

---

## 13.9 不建议作为独立主创新

- 普通 execution feedback；
- 普通 multi-agent；
- “加 Planner”；
- 大型 RAG；
- Best-of-N；
- 固定 early stopping；
- full profiler injection；
- 在线 Host/Kernel 双边 probe；
- 单纯增加 rollout/token budget。

---

## 13.10 最推荐实施顺序

```text
1. Frozen AscendC Base Agent
        ↓
2. Greedy/Fix-Loop
        ↓
3. Plan/Execute State Machine
        ↓
4. Plan Once / Every Step / Periodic / Stagnation
        ↓
5. Conditional Reasoning / Plan Amortization
        ↓
6. Patch Executor
        ↓
7. Selective Profiler
        ↓
8. Optional Host/Kernel case study
        ↓
9. Optional OpenEvolve / K-Search port
```

这个顺序的最大优点是：

> **每一步都直接检验一个成本来源，不需要为了“选择优化方向”额外付出大量 counterfactual experiments。**


---