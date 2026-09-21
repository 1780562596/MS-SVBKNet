# MS-SVBKNet

PyTorch code for **A dynamic scene image deblurring method based on multi-scale spatial variation blur kernel modeling and reliable reblurring consistency.**

本仓库提供论文 MS-SVBKNet 的训练、评测、图像推理、空间变化模糊核可视化和复杂度统计代码。模型联合使用多尺度退化表示（MDRM）、空间变化模糊核估计（SVBKM）、可学习基核字典和可信重模糊一致性（CRCM），实现动态场景图像去模糊。

## 论文结果与资源

下表为论文表 2 报告的结果。

| 数据集 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| GoPro | 31.02 dB | 0.941 | 0.073 |
| GS-Blur | 30.25 dB | 0.933 | 0.080 |



## 环境安装

使用 Python 3.10 或 3.11。正式训练建议使用支持 CUDA 的 GPU；CPU 可运行最小流程检查。先安装与本机环境匹配的 PyTorch 和 torchvision，再安装项目依赖：

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell 使用：.venv\Scripts\Activate.ps1
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -m pip install -e .
```

以上命令均在仓库根目录执行。CUDA 训练前运行 `python -c "import torch; print(torch.cuda.is_available())"`，输出应为 `True`。Windows 若遇到多进程数据加载问题，可添加 `--set data.num_workers=0`。

首次启用 LPIPS 时可能需要下载 VGG16 权重，请提前准备网络或权重缓存。正式配置启用 LPIPS；最小检查配置关闭 LPIPS，以便无需下载感知模型权重即可运行。YOLO 对比为可选功能，需要额外安装 `ultralytics`。

## 快速检查

```bash
python tools/make_toy_dataset.py
python train.py --config configs/smoke.yaml
python evaluate.py --config configs/smoke.yaml --checkpoint runs/smoke/seed_7/checkpoints/best.pt --output runs/smoke/seed_7/test
python -m pytest -q
```

`smoke.yaml` 使用小模型、CPU 和合成图像，仅用于检查训练和评测流程。正式实验使用下面的完整配置。

## 数据准备

### GoPro

按原始目录保留场景名称和模糊—清晰配对关系：

```text
GOPRO_Large/
├── train/
│   └── GOPRxxxx_xx_xx/
│       ├── blur/000001.png
│       ├── blur_gamma/000001.png
│       └── sharp/000001.png
└── test/
    └── GOPRxxxx_xx_xx/
        ├── blur/000001.png
        ├── blur_gamma/000001.png
        └── sharp/000001.png
```

官方训练分区包含 22 个场景、2,103 对图像，测试分区包含另外 11 个场景、1,111 对图像。论文将训练分区进一步分为 **18 个训练场景和 4 个验证场景**，划分种子为 3407；测试集仅用于最终评估。

`configs/full.yaml` 读取 `blur_gamma`，`configs/full_blur.yaml` 读取 `blur`。一次实验只使用一种模糊版本。先将所用 YAML 的 `data.root` 改为本机数据目录，例如 `D:/datasets/GOPRO_Large`。

在尚无论文固定 CSV 的情况下，可按上述场景数量生成本地划分。使用 `--val-scenes 4` 指定 4 个验证场景；数量相同不代表与论文原始清单逐项相同：

```bash
python tools/build_scene_split.py --root /path/to/GOPRO_Large --blur-variant blur_gamma --val-scenes 4 --seed 3407
python tools/audit_dataset.py --config configs/full.yaml --output data_audit.json
```

索引保存至数据根目录的 `train_index.csv` 与 `val_index.csv`。如运行 `blur` 配置，使用对应的独立索引文件名：

```bash
python tools/build_scene_split.py --root /path/to/GOPRO_Large --blur-variant blur --val-scenes 4 --seed 3407 --train-output train_index_blur.csv --val-output val_index_blur.csv
python tools/audit_dataset.py --config configs/full_blur.yaml --output data_audit_blur.json
```

### GS-Blur 或自定义数据

使用 `configs/gs_blur.yaml`，设置 `data.root` 及三个索引路径。CSV 必须包含 `blur` 和 `sharp` 两列；相对路径以 `data.root` 为基准：

```csv
blur,sharp
train/scene_001/blur/0001.png,train/scene_001/sharp/0001.png
```

论文按重建场景划分 GS-Blur：训练 2,726 个场景、验证 341 个场景、测试 341 个场景。相同场景、相同清晰参考图及其不同运动、缩放和噪声变体必须归入同一分区。GoPro 与 GS-Blur 分别训练模型。

若数据已正确划分，且 blur/sharp 目录下文件相对路径一一对应，可分别生成索引：

```bash
python tools/build_pair_index.py --root /path/to/GS-Blur --blur-dir /path/to/GS-Blur/train/blur --sharp-dir /path/to/GS-Blur/train/sharp --output train_index.csv
python tools/build_pair_index.py --root /path/to/GS-Blur --blur-dir /path/to/GS-Blur/val/blur --sharp-dir /path/to/GS-Blur/val/sharp --output val_index.csv
python tools/build_pair_index.py --root /path/to/GS-Blur --blur-dir /path/to/GS-Blur/test/blur --sharp-dir /path/to/GS-Blur/test/sharp --output test_index.csv
```

该工具只负责按路径配对，不负责 GS-Blur 场景分组或一对多映射。复杂映射需要直接提供 CSV。GS-Blur 示例配置没有自动启用场景数量及场景身份校验，使用前须核对原始场景 ID 与共享参考图；详情见 [数据与评测协议](docs/EVALUATION.md)。

## 训练与断点续训

```bash
python train.py --config configs/full.yaml
python train.py --config configs/gs_blur.yaml
```

完整配置采用 256×256 裁剪、300 个 epoch、batch size 2、种子 3407、16 个 15×15 基核。生成器/判别器学习率为 `5e-5`，退化建模分支学习率为 `1e-4`，Adam 参数为 `(0, 0.9)`。损失权重及模型配置见 [实现说明](docs/IMPLEMENTATION_SPEC.md)。

命令行可覆盖配置，例如 `--set train.batch_size=1`。每次运行会在 `runs/<实验名>/seed_<种子>/` 中保存完整配置、实验环境、数据审计、训练日志、图像预览和检查点。`best.pt` 按验证集 PSNR 选择，`last.pt` 用于续训。

```bash
python train.py --config configs/full.yaml --set train.epochs=400 --resume runs/ms_svbknet_full/seed_3407/checkpoints/last.pt
```

续训会恢复模型、优化器、AMP、动态损失权重和随机状态。训练签名会拒绝架构、数据和 batch size 等状态相关变更；允许修改总训练轮数等非状态配置。同一软硬件环境中的确定性续训以 epoch 边界为单位。

## 评测与推理

完整测试集评测：

```bash
python evaluate.py --config configs/full.yaml --checkpoint runs/ms_svbknet_full/seed_3407/checkpoints/best.pt --output results/gopro --save-images
```

输出包括 `metrics.json`、`per_image.csv`，以及启用 `--save-images` 后的恢复图像。指标文件记录检查点 SHA-256、数据配对哈希和评测协议。

单图或目录推理：

```bash
python infer.py --config configs/full.yaml --checkpoint /path/to/best.pt --input /path/to/blurry_images --output results/restored
```

使用与权重匹配的配置。默认 `eval.tile=0` 处理整幅图像；显存不足时可增加 `--set eval.tile=512`，分块结果应单独标明。目录推理保留相对路径。

正式配置采用完整 RGB、crop 0、8 位量化 PSNR/SSIM，以及官方校准 LPIPS v0.1 / VGG16。`float_rgb` 为另一种显式协议，不能与量化结果混合比较。详细定义见 [数据与评测协议](docs/EVALUATION.md)。

## 可选分析工具

```bash
# 多随机种子训练；固定数据划分
python run_repeats.py --config configs/full.yaml --seeds 3407 3408 3409 --name full
# 核、空间权重及可靠性图
python visualize_kernels.py --config configs/full.yaml --checkpoint /path/to/best.pt --image /path/to/example.png --output results/kernels.png
# 当前配置的参数量、计算量和运行速度
python complexity.py --config configs/full.yaml --size 256 --device cuda
# 下游车辆检测
python -m pip install ultralytics
python tools/yolo_compare.py --blur-dir /path/to/blur --restored-dir results/restored
```

多种子工具输出逐次结果及均值/样本标准差；论文主实验使用种子 3407。复杂度和速度以当前配置、硬件上的实测输出为准。

YOLO 工具默认统计模糊输入与恢复输出的车辆数量、平均置信度。提供 `--blur-data` 与 `--restored-data` 两个带真实标注的数据集 YAML 后可计算 mAP、Precision、Recall。添加 `--sharp-dir /path/to/sharp` 和 `--sharp-data /path/to/sharp.yaml` 可运行清晰参考分支。各组目录须包含相同相对路径的图像；带标注评测需要每组都提供 YAML 并指向同一组帧及共享标注。工具使用论文参数：`imgsz=640`、`conf=0.25`、`val-conf=0.001`、`iou=0.70`。

消融定义及当前代码支持范围见 [消融说明](docs/ABLATION.md)。

## 仓库结构

```text
configs/             完整训练及最小检查配置
ms_svbknet/models/   生成器、MDRM、SVBKM、判别器
ms_svbknet/          数据加载、损失、训练引擎、评测与工具函数
tools/               数据索引、审计、合成数据、YOLO 对比
tests/               模型与实验协议检查
docs/                实现、评测、消融及验证说明
train.py             训练入口
evaluate.py          测试入口
infer.py             单图及目录推理入口
run_repeats.py       多随机种子实验
visualize_kernels.py 模糊核与可靠性可视化
complexity.py        参数量、计算量与速度统计
```

## GitHub 发布

按公开仓库 `1780562596/MS-SVBKNet` 准备；尚未上传。需要补充的信息和具体提交步骤见 [GitHub 发布说明](docs/GITHUB_RELEASE.md)。

## 引用与许可

论文标题见本页开头。作者、发表信息、论文链接及正式 BibTeX 将在信息确认后补充。

本版本尚未附带开源许可证，具体授权方式待作者确定。数据集、第三方依赖及其权重遵循各自的许可条款。

## 相关资源

- [GoPro 数据集](https://seungjunnah.github.io/Datasets/gopro.html)
- [GS-Blur](https://github.com/dongwoohhh/GS-Blur)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity)
- [DeblurGAN-v2](https://github.com/VITA-Group/DeblurGANv2)、[Learning Degradation Representations](https://github.com/dasongli1/Learning_degradation)、[ASPDC](https://github.com/Dong-Huo/ASPDC)
- [MIMO-UNet](https://github.com/chosj95/MIMO-UNet)、[Restormer](https://github.com/swz30/Restormer)、[DeepDeblur-PyTorch](https://github.com/SeungjunNah/DeepDeblur-PyTorch)
