实现 Experiment Knowledge Loop。


新增：

IncidentRecord。


记录：

- failure
- hypothesis
- code diff
- resolved facts
- result


规则：

只有：

correctness通过

failure消失

才允许生成：

ConfirmedExperience。


禁止：

experience覆盖official。


增加：

frontier管理：

source frontier

compile frontier

runtime frontier

correctness frontier

performance best


失败candidate不得污染下一轮baseline。


验收：

连续失败实验：

可以自动回滚。

成功实验：

可以形成经验记录。

