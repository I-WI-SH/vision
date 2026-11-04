# 项目3.0 peft 视觉部分

## 1. 环境配置

### 1.1 创建 Conda 环境

```bash
conda create -y -n dassl python=3.9 --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
```

创建完成后，激活环境：

```bash
conda activate dassl
```

### 1.2 安装 PyTorch 和 torchvision

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

### 1.3 安装 Dassl_pytorch 依赖

在 `vision/Dassl_pytorch` 目录下执行：

```bash
cd vision/Dassl_pytorch
python -m pip install -r requirements.txt -i https://pypi.org/simple (https://mirrors.aliyun.com/pypi/simple)
python setup.py develop
```

### 1.4 安装 其他 依赖

完成上一步后，返回 `vision` 目录并执行：

```bash
cd vision/
python -m pip install -r requirements.txt -i https://pypi.org/simple
python -m pip install -e ./auto-attack
```

## 2. 准备遥感数据集

下载 [EuroSAT_RGB 数据集](https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip)。

python -m zipfile -e ./DATA/EuroSAT_RGB.zip ./DATA/eurosat/
数据目录如下
```
eurosat/
|–– 2750/
|–– split_zhou_EuroSAT.json
```

## 3. 准备模型

通过 hfd 下载 `TreeLLi/Adversarial-Prompt-Tuning` 仓库中的模型权重 [vitb32_eps4.pth.tar](https://huggingface.co/TreeLLi/Adversarial-Prompt-Tuning/blob/main/backbone/vitb32_eps4.pth.tar)。

准备好 `hfd.sh` 及其所需依赖后，在 `vision` 目录下执行（`hfd.sh` 位于当前目录）：

```bash
bash hfd.sh TreeLLi/Adversarial-Prompt-Tuning --include "backbone/vitb32_eps4.pth.tar" --local-dir ./backbone
```

## 4. 适配任务
搭建好环境、下载数据集。
下载好模型以后需要注意替换trainers/apt.py; fap.py; spt.py 中 /home/ubuntu/pengguowei/backbone的地址。

适配任务是在910B 上可以进行scripts下面3个脚本任务的训练