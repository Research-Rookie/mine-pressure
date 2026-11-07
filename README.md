# 矿压预测模型

本仓库包含矿压预测的网络结构、历史特征、训练目标、概率校准、区域聚合和均值协调代码。

```bash
python -m pip install -r requirements.txt
```

| 模块 | 用途 |
| --- | --- |
| `minepressure.preprocessing` | 组织历史输入，拟合训练集预处理 |
| `minepressure.network.MinePressureNetwork` | 组装数值预测、概率分布和预警分支 |
| `minepressure.training` | 训练目标、采样与优化步骤 |
| `minepressure.calibration` | 概率校准和报警阈值选择 |
| `minepressure.regional_maximum_distribution` | 区域最大值分布 |
| `minepressure.tail_preserving_normal_projection` | 上尾保持与可达均值协调 |

## 调用接口

`prepare_history_arrays` 接收历史监测与工况数组。调用方传入工程上限和推进量缩放尺度，并根据实际数据可用时间截取历史窗口。输入监测数组的维度顺序为样本、历史班次、区域、监测通道。

构造 `MinePressureNetwork` 时传入数值分支统计量、分布区间边界、主体特征均值与尺度、预警原型初始状态及两个分布分支的宽度。调用方可以通过 PyTorch `load_state_dict` 加载自己的权重。

推理接收 `sequence`、`summary_raw`、`anchor_raw` 和 `installed`。预处理对象、概率校准系数、原始得分报警阈值、工程上下限及批大小由调用方传入。`probability_links` 的两列分别是斜率和截距。`raw_cuts` 按预警头顺序提供阈值。没有下限的通道可传入负无穷。

预处理统计量应由训练集拟合，报警阈值与概率校准系数应由校准集确定。评估时使用对应的有效标签掩码。工作面聚合要求每个输出参数至少有一个有效区域。
