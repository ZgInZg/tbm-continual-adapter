# 盾构多工况持续学习课程作业代码说明

本目录为《高级机器学习理论》课程报告
《基于共享状态空间与轻量适配器的盾构多工况持续学习方法》
对应的可提交代码与数据。

## 1. 目录结构

- `config.py`
  统一保存数据路径、工况名称、共享变量、目标变量与公共超参数。
- `shared_utils.py`
  提供数据读取、连续环样本构造、线性基础模型训练、轻量适配器训练与评价函数。
- `01_shared_structure_discovery.py`
  共享结构识别主实验：PCMCI 稳定候选结构发现 + PCMCI+ 严格方向复核。
- `02_shared_structure_sensitivity.py`
  共享结构对窗口比例和稳定阈值的敏感性分析。
- `03_condition_shift_validation.py`
  工况分布差异验证：PCA、Wasserstein 距离、工况可辨识性分析。
- `03b_condition_shift_validation_zh.py`
  与 `03` 对应的中文绘图版本。
- `04_continual_adaptation_main.py`
  多工况持续适配主实验。
- `05_lowshot_s3_adaptation.py`
  S3 小样本适配实验。
- `data/`
  输入样例数据。
- `results/`
  各脚本运行后生成的结果目录。
- `requirements.txt`
  Python 依赖列表。

## 2. 数据说明

考虑到工程数据的保密要求，仓库**不公开完整原始数据**，而是提供一套
**最小可运行的脱敏样例**：

- `data/sheet1_cycle_duration_modeling_shrunk.csv`
- `data/new_condition_ring_level_mean.csv`

这两份样例数据仅保留了课程作业主线所需的最小字段和少量连续环样本，用于：

- 验证代码流程可以正常运行；
- 说明输入数据格式；
- 展示共享状态空间建模所需的字段组织方式。

说明：

- 报告中使用 `S1`、`S2`、`S3` 表示三个工况。
- 原始主表中的工程编号为 `S908`、`S909`，代码中已统一映射为 `S1`、`S2`，与报告保持一致。
- 公开样例中仅保留课程作业主线所需的三个共享变量：
  `pressure_state`、`total_thrust`、`cutterhead_torque`。
- 仓库中的公开样例**不能完全复现报告中的全部数值结果**，因为报告后半部分 `5.4`
  涉及的 `138` 个 S3 私有操作参数属于工程敏感数据，未在公开仓库中提供。
- 因此，公开仓库主要用于复现报告的主流程与 `5.1-5.3` 部分；`5.4` 的扩展消融保留在报告中，
  但对应的完整输入数据不公开。

## 3. 环境配置

建议使用 Python `3.10` 或 `3.11`。

安装依赖：

```bash
pip install -r requirements.txt
```

如果本地没有 `tigramite` 或 `matplotlib`，请先安装完成再运行脚本。

## 4. 推荐运行顺序

```bash
python 01_shared_structure_discovery.py
python 02_shared_structure_sensitivity.py
python 03_condition_shift_validation.py
python 04_continual_adaptation_main.py
python 05_lowshot_s3_adaptation.py
```

运行后输出会自动写入 `results/` 下对应编号的子目录。

## 5. 各脚本输出内容

- `01`：
  输出共享候选边、PCMCI+ 严格有向边、同期耦合邻接与连续环段统计。
- `02`：
  输出窗口比例与稳定阈值敏感性表。
- `03`：
  输出 PCA 图、成对 Wasserstein 距离与工况分类结果。
- `04`：
  输出持续适配主实验总表、分任务表和综合排名表。
- `05`：
  输出 S3 小样本适配结果、不同观测比例下的方法比较表。

## 6. 与报告结果的对应关系

- 共享结构识别结果：对应报告第 `5.1` 节。
- 工况分布差异验证：对应报告第 `4.3` 节和图 `2`。
- 持续适配主实验：对应报告第 `5.2` 节。
- S3 小样本适配：对应报告第 `5.3` 节。

## 7. 提交说明

本仓库已包含：

- 可运行代码；
- 脱敏后的输入样例；
- 清晰的运行流程说明。

若需要与课程报告对照阅读，请结合仓库中的 `report/高级机器学习.docx` 一并查看。
