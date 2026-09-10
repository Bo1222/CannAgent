在 Phase0 完成基础上，实现离线结构化知识构建与发布。

禁止修改：

- planner
- generator
- evaluator

只修改：

structured_knowledge/


目标：

建立：

CANN docs

↓

Published Knowledge Build


实现：

knowledge_store/

cann/
 8.5.0/
   current.json
   builds/
      knowledge_build_id/

        raw/
        normalized/
        facts/
        cards/
        indexes/


要求：

1. 已发布的知识构建不可变。

构建流程：

build staging

↓

validate

↓

publish knowledge build and update current.json


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

生成并发布 CANN 8.5 结构化知识

包含：

- DataCopy
- DataCopyPad
- TPipe
- TQue


测试：

knowledge build load成功
source provenance正确
重复build结果一致
