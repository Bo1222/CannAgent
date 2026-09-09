实现 Semantic Validator。


禁止：

硬编码大量API规则。


新增：

CallSemanticsResolver。


目标：

代码调用：

DataCopyPad(...)


解析：

API

overload

parameter binding


然后查询：

Knowledge Snapshot。


输出：

ResolvedApiCall。


例如：

{
api:
DataCopyPad,

parameter:
blockLen,

unit:
byte,

source_fact:
fact_id
}


Validator检查：

- 单位
- alignment
- memory scope
- stream contract
- tiling constraints


所有规则必须来自：

AtomicFact

ProjectContract


禁止：

if api=="DataCopyPad":

这种规则。


验收：

错误：

blockLen单位错误

必须在compile前发现。