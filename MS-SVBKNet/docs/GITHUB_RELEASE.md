# GitHub 发布准备

目标账号：`1780562596`。建议仓库名：`MS-SVBKNet`。可见性：公开。该地址是拟定目标，本文不表示远程仓库已经创建或上传成功。

建议仓库简介：

> PyTorch code for MS-SVBKNet: dynamic scene image deblurring with spatially varying blur kernels and reliable reblurring consistency.

## 发布前需要补充

1. 作者姓名、顺序、单位，以及论文链接或 DOI；据此补充正式引用。
2. 作者选择的许可证；当前未代为添加 MIT、Apache-2.0 等授权。
3. GoPro 与 GS-Blur 固定 train/val/test CSV、场景分组说明和对应哈希。
4. 两个数据集的训练权重、对应完整配置、评测输出和下载地址。
5. 正式实验的 PyTorch/CUDA、GPU、数据版本、量化与 SSIM 设置。
6. 确认生成器与 MDRM 的融合连接、通道数和残差激活细节，见 IMPLEMENTATION_SPEC.md。
7. 如需公开全部扩展实验，补充完整消融配置、50 张合成核评测清单及交通图像标注。

## 提交内容

提交源代码、configs、tests、docs、README、依赖文件、.gitignore、.gitattributes 与 .github。模型权重、数据集、运行输出和本机环境不纳入普通 Git 提交。大体积权重使用独立下载或 GitHub Release 附件。

## 操作步骤

先在账号 1780562596 下创建名为 MS-SVBKNet 的空公开仓库。若使用下面的初始化流程，创建时不要自动添加 README 或许可证。确认在本项目根目录后执行：

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Prepare MS-SVBKNet paper code"
git remote add origin https://github.com/1780562596/MS-SVBKNet.git
git push -u origin main
```

若已存在 Git 仓库或 origin，应检查当前状态后使用对应流程。首次提交使用自己的 Git 姓名和邮箱；不要把访问令牌写入文件。成功发布后，再把实际仓库地址补入论文 Code and Data Availability。
