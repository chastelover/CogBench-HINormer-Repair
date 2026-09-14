# CogBench V5.2 官方 HINormer 单行修复与整套流程恢复

本包只处理当前唯一未决运行：

- `data_seed=269`
- `alpha=0.9`
- `train_seed=1003`
- 当前父运行：`result_max_epochs=5000`

它不会修改 CogBench 项目源码，也不会改动原始 `protocol.json` 中的 5000 轮基础协议签名。

## 为什么需要继续训练

当前运行不是预测坍缩：`degenerate_predictions=0`、正类预测率为 0.518、概率标准差为 0.335734。它未通过平台期认证的唯一原因是最后 120 轮验证集 AP 跨度为 `0.001705`，高于冻结阈值 `0.001`。因此不能人工改标记或放宽阈值；应按预先固定的 8000、12000 轮阶梯继续验证。

## 一键启动

把压缩包解压到 `/CogBench` 后执行：

```bash
cd /CogBench/CogBench_V5_2_Official_HINormer_Repair_v1
bash start_repair_and_resume.sh
```

不需要手动激活 conda 环境。脚本默认使用：

- 核心环境：`/usr/local/miniconda3/envs/py312/bin/python`
- 官方 HINormer 环境：`/usr/local/miniconda3/envs/cogbench-hinormer-cpu/bin/python`
- 项目：`/CogBench/CogBench_Final_Experiments_V5_2`
- 输出：`outputs/final_paper_v5_2`
- Plateau-v2 恢复器：`/CogBench/CogBench_V5_2_Plateau_V2_Recovery`

## 查看状态

```bash
cd /CogBench/CogBench_V5_2_Official_HINormer_Repair_v1
bash status_repair_and_resume.sh
```

持续看日志：

```bash
tail -f /CogBench/CogBench_V5_2_Official_HINormer_Repair_v1/repair_and_resume.log
```

断开网页终端不会终止任务；启动器使用 `nohup` 在后台完成修复并继续原来的整套 canonical suite。

## 成功标准

最终日志应同时出现：

```text
OFFICIAL_HINORMER_EXTENSION=PASS selected_cap=8000|12000
OFFICIAL_RESUME_VALIDATION=PASS retained=150
FINAL_VERIFY=PASS ... strict_gate=PASS blockers=0 claim_ready=True claim_failures=0
```

最终结果仍以以下文件为准：

```text
/CogBench/CogBench_Final_Experiments_V5_2/outputs/final_paper_v5_2/final_release_report.json
```

## 审计与安全约束

1. 8000 轮必须逐字节复现已保存的前 5000 轮历史；若进入 12000 轮，还必须逐字节复现刚得到的前 8000 轮历史。
2. 是否进入下一预算只看验证集 AUC、AP、loss 和学习率；选定最终预算前不会读取测试集指标。
3. 测试集只在最终预算确定后评估一次。
4. 候选结果先写入隔离目录；通过全部校验后才事务式替换唯一目标行。
5. 任一写入或校验失败会自动回滚；下次启动也会先恢复未完成事务。
6. 其余 149 行的训练历史、预测、预测清单和 checkpoint 必须保持不变。
7. 最终门禁要求每个扩展训练记录 `parent_max_epochs`。现有适配器漏写了这个字段，因此脚本会为其他已认证的 capped/non-early 行做可验证的元数据回填；不会改变其指标、预测或训练历史。
8. 修复以独立、自哈希的 protocol amendment 记录，不篡改 5000 轮基础运行签名。

## 如果 12000 轮仍未解决

脚本会显示 `UNRESOLVED_AT_TERMINAL_CAP` 并退出，且不会修改正式结果。此时不要继续无限增加 epoch，也不要放宽阈值；应在论文中如实报告该边界运行，或将官方 HINormer 从主表降为补充实验。

## 自定义路径（仅路径不同才需要）

```bash
COGBENCH_PROJECT=/your/project \
COGBENCH_OUTPUT_ROOT=/your/output \
COGBENCH_PLATEAU_V2_RECOVERY=/your/recovery \
COGBENCH_OFFICIAL_PYTHON=/your/official/python \
bash start_repair_and_resume.sh
```

预算阶梯和目标 key 在本版本中被冻结，不能通过环境变量修改。
