# 验证记录

检查日期：2026-09-20。环境：Windows，Python 3.12，PyTorch 2.6.0+cpu，torchvision 0.21.0+cpu。临时测试依赖放置在仓库外，不随代码发布。

## 本次执行结果

- 模型、数据协议及工具测试：**27 passed**。
- 合成小数据 CPU 训练：1 个 epoch，1 次训练迭代，成功保存 best/last 检查点。
- 使用 best 检查点评测：成功，smoke 配置限制为 1 张测试图。
- 目录推理：成功处理 4 张图并保留相对路径。
- 数据审计：通过，合成集 train/val/test 为 4/2/2 个场景和配对。
- 断点续训：从第 1 个 epoch 成功继续至第 2 个 epoch。
- 完整模型 256×256 前向计算与复杂度工具：通过。

新增测试验证 18/4 划分、确定性场景选择、非法划分参数、跨验证/测试场景泄漏、旧 20/2 分区拒绝，以及三组 YOLO 参数一致性。YOLO 接口测试使用替身检测器检查调用协议，不代表真实交通数据已完成评测。

## 完整模型测量

`configs/full.yaml`，输入 1×3×256×256：

| 项目 | 当前代码测量 |
|---|---:|
| 参数量 | 11,548,203 |
| THOP MACs | 60.779266048 G |
| 按 2×MACs 计 FLOPs | 121.558532096 G |

参数量约 11.55M。论文表 3 的 FLOPs 为 121.34G，当前测量约 121.56G；应核对论文使用的模型版本、分析器版本和统计范围。此处没有将测量结果调整为论文数字。

复杂度统计包含生成器、MDRM 与 SVBKM，不包含训练判别器、重模糊操作和 CRCM。CPU 单次计时仅用于功能检查，不能替代论文 GPU 延迟/FPS/显存。

## 检查范围

本次没有进行 GoPro/GS-Blur 的完整训练，未加载论文正式权重，也未验证论文精度数字。smoke 配置关闭 LPIPS；LPIPS 配置和接口已通过单元测试，但本次未下载并运行真实 VGG16 校准权重。未执行真实 YOLO 交通数据评测。

GitHub Actions 已配置 Python 3.10/3.11 的 CPU 测试和最小流程；远程工作流尚未运行。

## 本机重复检查命令

```bash
python -m pytest -q
python tools/make_toy_dataset.py
python train.py --config configs/smoke.yaml
python evaluate.py --config configs/smoke.yaml --checkpoint runs/smoke/seed_7/checkpoints/best.pt --output runs/smoke/seed_7/test
python infer.py --config configs/smoke.yaml --checkpoint runs/smoke/seed_7/checkpoints/best.pt --input data/toy/test --output runs/smoke/inference
python tools/audit_dataset.py --config configs/smoke.yaml --output runs/smoke/audit.json
python train.py --config configs/smoke.yaml --set train.epochs=2 --resume runs/smoke/seed_7/checkpoints/last.pt
python complexity.py --config configs/full.yaml --device cpu --warmup 1 --repeats 1
```
