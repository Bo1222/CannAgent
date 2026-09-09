基于已经存在的 Knowledge Snapshot。

实现 KnowledgeRouterV2。


禁止修改：

- generator prompt
- planner逻辑


只修改：

knowledge router

context builder


目标：

将：

doc retrieval

升级为：

semantic knowledge retrieval。


输入：

KnowledgeContext:

包含：

- operator
- phase
- CANN version
- SoC
- source symbols
- failure(optional)
- active plan


输出：

KnowledgeBundle


KnowledgeBundle包含：

- API semantics
- relevant facts
- examples
- failure cards
- project contracts
- provenance


检索顺序：

1 exact API
2 overload/context
3 metadata
4 FTS
5 vector


禁止：

embedding直接决定API选择。


增加：

retrieval trace。


记录：

选择了什么
为什么选择
为什么拒绝


验收：

同一个：

DataCopy

DataCopyPad

DataCopyExt


不会混淆。


legacy router保持可用。


新增：

--knowledge-mode legacy|semantic