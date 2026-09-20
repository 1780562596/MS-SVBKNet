# 消融实验说明

论文第 4.6 节包含表 8、9、10 三组实验。本文件给出定义与代码支持范围，实验结果应由相应配置实际运行记录。

## 表 8 累积模块实验

所有行均使用 **10 个三卷积改进残差块**，不是传统两卷积残差块。

| 行 | MDRM | SVBKM / 字典 | CRCM | 判别器 |
|---|---|---|---|---|
| Baseline | 无 | 无 | 无 | Standard PatchGAN |
| +MDRM | 有 | 无 | 无 | Standard PatchGAN |
| +SVBKM | 有 | 空间权重 + 16 个固定基核 | 无 | Standard PatchGAN |
| +Complete SVBKM | 有 | 空间权重 + 可学习字典 | 无 | Standard PatchGAN |
| +CRCM | 有 | 完整 SVBKM | 有 | Standard PatchGAN |
| Full MS-SVBKNet | 有 | 完整 SVBKM | 有 | Improved PatchGAN |

最后两行仅改变判别器。当前发布代码的训练入口使用 Improved PatchGAN，未提供 Standard PatchGAN 的完整训练配置，不能用简单模块开关代替整张表。

## 表 9 单组件实验

包括去掉 MDRM、空间变化机制、可学习字典、CRCM，以及传统残差块、取消分层 Dropout、4/7/12 个残差块。

当前可通过 `dictionary_mode=fixed` 冻结字典，通过 `use_crcm=false` 关闭可信损失。生成器残差块数和结构在代码中固定；去掉 MDRM 后 SVBKM 的替代输入、取消空间变化后的全局核形式，论文未给出完整接口定义。相应消融训练代码与配置尚未包含。

## 表 10 基核数与尺寸

基核数实验固定尺寸 15×15，数量为 4/8/16/32；尺寸实验固定 16 个基核，尺寸为 7/11/15。

```bash
python train.py --config configs/full.yaml --set model.num_kernels=8 --set experiment.name=kernel_count_8
python train.py --config configs/full.yaml --set model.kernel_size=11 --set experiment.name=kernel_size_11
```

每组应使用独立输出目录，并以同一组覆盖参数评测对应检查点。16 个 15×15 基核是完整模型配置。论文该配置的 GoPro 指标为 31.02 / 0.941 / 0.073。

## 实验控制

各组固定数据划分、种子 3407、训练预算和评测协议；只改变该组定义的因素。保存完整配置、最佳 epoch、权重与数据哈希。多种子实验工具是附加分析功能，论文当前主实验描述为固定种子。
