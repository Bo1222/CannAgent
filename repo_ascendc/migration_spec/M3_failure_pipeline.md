实现 Structured Failure 系统。


禁止修改：

knowledge build


修改：

evaluator
diagnostics
trajectory


目标：

将：

runtime log

转换为：

StructuredFailure。


Schema：

StructuredFailure:

{
stage,

runtime_code,

subsystem,

device_exception,

reason,

core_id,

block_id,

sub_error_type,

case_info,

related_symbols
}


支持：

- MTE
- AICORE exception
- ACL error
- compile error


要求：

禁止只保存：

error_excerpt


必须保存：

结构化字段。


然后接入 StructuredKnowledgeRouter。


流程：

runtime failure

↓

StructuredFailure

↓

FailureCard retrieval

↓

Diagnose prompt


验收：

给MTE日志：

必须产生：

{
subsystem:"MTE",
reason:"illegal configuration"
}
