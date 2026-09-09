在 Phase0 完成基础上，实现 Offline Knowledge Snapshot。

禁止修改：

- planner
- generator
- evaluator

只修改：

knowledge_v2/


目标：

建立：

CANN docs

↓

Knowledge Snapshot


实现：

knowledge_store/

cann/
 8.5.0/
   snapshots/
      snapshot_id/

        raw/
        normalized/
        facts/
        cards/
        indexes/


要求：

1. Snapshot immutable。

构建流程：

build staging

↓

validate

↓

publish snapshot


2. 增加：

ApiCard

FailureCard

PatternCard


3. 实现：

FactValidator


检查：

- fact必须有evidence
- fact必须有applicability
- fact必须有source hash


4. 实现：

ConflictResolver


禁止：

简单覆盖。


只有：

same API
same overload
same version
same context

出现冲突才报告。


5. 增加CLI：

python -m ascendc_multi_turn.knowledge.build


验收：

生成CANN8.5 snapshot

包含：

- DataCopy
- DataCopyPad
- TPipe
- TQue


测试：

snapshot load成功
source provenance正确
重复build结果一致