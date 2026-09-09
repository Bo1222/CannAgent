你正在改造 AscendC Kernel Agent 的知识系统。

当前阶段只允许实现 Offline Knowledge Compiler MVP。

禁止修改：
- generator
- planner
- diagnose
- evaluator
- runtime router
- source validator

禁止改变现有 Agent 行为。

目标：

证明 CANN 官方文档可以被转换为结构化知识。

实现：

official docs
    |
    v
Document Parser
    |
    v
NormalizedDocument
    |
    v
AtomicFact
    |
    v
Query API


具体任务：

1. 新增：

ascendc_multi_turn/knowledge_v2/

结构：

knowledge_v2/
├── schema.py
├── normalize.py
├── extract.py
├── query.py
├── tests/


2. 实现 Schema：

NormalizedDocument

包含：

- document id
- source path
- title
- heading tree
- paragraphs
- tables
- code candidates
- constraints
- examples


AtomicFact

必须包含：

- subject
- predicate
- value
- applicability
- provenance


3. 实现 deterministic markdown parser。

要求：

- 不调用 LLM
- 不解释语义
- 不生成事实

只负责：

Markdown
→
结构化节点


4. 实现 FactExtractor interface。

第一阶段可以 mock。

接口：

extract(normalized_document)

返回 AtomicFact。


5. 所有 fact 必须带 provenance：

包括：

- source document
- section
- evidence text


6. 增加测试：

必须证明：

DataCopyPad 文档

可以生成：

{
 api:
 DataCopyPad,

parameter:
 blockLen,

source:
 section_xxx
}


验收：

- 不修改任何 runtime 文件
- 原测试全部通过
- 可以查询 DataCopyPad相关事实


输出：

修改文件列表
新增测试
后续阶段建议