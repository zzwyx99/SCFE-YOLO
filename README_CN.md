# UAV_yolo12

基于 Ultralytics YOLO 的无人机小目标检测实验仓库，当前项目主要围绕 `ScaleConsistencyCoupledSAEM` 及其消融版本展开，数据集以 VisDrone 为主。

项目核心文件：

- `custom_blocks.py`：自定义模块与消融模块
- `configs/*.yaml`：不同模型结构与消融配置
- `trainSCFEYOLO.py`：训练入口，先注册自定义模块，再交给 Ultralytics 构建模型
- `tools/manual_flops.py`：手动统计参数量与 FLOPs

## 1. Environment

建议使用独立的 Conda 环境。

推荐新建环境：

```bash
conda create -n scfe-yolo python=3.8 -y
conda activate scfe-yolo
``
安装 PyTorch。若使用 GPU，请按 PyTorch 官方页面选择与你本机 CUDA 对应的命令：

```bash
pip install torch==2.4.1 torchvision==0.19.1
```
安装 Ultralytics 与常用依赖：

```bash
pip install ultralytics==8.4.22
pip install opencv-python==4.12.0.88 pandas==2.0.3 scipy==1.10.1 tqdm==4.67.1 matplotlib
```

## 2. Dataset

当前仓库中已有 `datasets/VisDrone/` 目录，包含图像、标签和原始压缩包。训练时默认使用：

```python
model.train(data="VisDrone.yaml", ...)
```

如果你的环境里还没有 `VisDrone.yaml`，可以按 Ultralytics 检测数据格式自行创建，例如：

```yaml
path: ./datasets/VisDrone
train: images/train
val: images/val
test: images/test

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

如果你的类别定义与标签编号不同，请按自己的数据实际情况修改 `names`。

## 3. Quick Start

训练前需要先注册自定义模块，否则 Ultralytics 的模型解析器无法识别 `configs/*.yaml` 中的自定义层名。当前训练脚本已经处理好这一步：

```python
from ultralytics import YOLO
from custom_blocks import register_custom_modules

register_custom_modules()
model = YOLO("configs/NSR.yaml")
```

直接运行：

```bash
python trainSCFEYOLO.py
```

默认训练入口见 [trainSCFEYOLO.py](trainSCFEYOLO.py)。

如果你想切换到其它结构，可以直接修改：

```python
model = YOLO("configs/SCFE-YOLO.yaml")
model = YOLO("configs/NSRBR.yaml")
model = YOLO("configs/NSRBRCC.yaml")
model = YOLO("configs/wobranchrouter.yaml")
model = YOLO("configs/nospatialgate.yaml")
model = YOLO("configs/noFE.yaml")
```

## 4. Configs

当前仓库内主要配置如下：

- `configs/SCFE-YOLO.yaml`
  完整模型，使用 `ScaleConsistencyCoupledSAEM`
- `configs/NSR.yaml`
  仅保留 NSR
- `configs/NSRBR.yaml`
  NSR + BR
- `configs/NSRBRCC.yaml`
  NSR + BR + CSP
- `configs/wobranchrouter.yaml`
  去掉 Branch Router
- `configs/nospatialgate.yaml`
  去掉 DSG / Spatial Gate
- `configs/noFE.yaml`
  去掉 native enhancement，仅保留跨尺度耦合路径
- `configs/nop5.yaml`
  去掉最深层 P5 stage
- `configs/AF-YOLO.yaml`
  仓库中保留的另一套 AF-YOLO 风格结构

## 5. Custom Blocks

自定义模块定义在 [custom_blocks.py](custom_blocks.py) 中，当前主要包含：

- `ScaleAwareEdgeMixer`
  基础版多分支纹理/上下文/边缘混合模块
- `ScaleConsistencyCoupledSAEM`
  完整版 SCC-SAEM，包含 NSR、BR、CSP、CGA 和 DSG
- `ScaleConsistencyCoupledSAEMNSR`
  仅保留 native-scale reference construction
- `ScaleConsistencyCoupledSAEMNSRBR`
  NSR + BR
- `ScaleConsistencyCoupledSAEMNSRBRCSP`
  NSR + BR + CSP
- `ScaleConsistencyCoupledSAEMNSRBRCSPCGA`
  NSR + BR + CSP + CGA
- `ScaleConsistencyCoupledSAEMNoBranchRouter`
  去掉分支路由器
- `ScaleConsistencyCoupledSAEMNoSpatialGate`
  去掉空间门控
- `ScaleOnlyCoupledSAEM`
  去掉 native enhancement，仅保留 scale consistency 路径

缩写说明：

- `NSR`：native-scale reference construction
- `BR`：branch router
- `CSP`：compressed-scale perturbation
- `CGA`：consistency-gated alignment
- `DSG`：discrepancy-aware spatial gate

## 6. Notes

- 使用 Ultralytics CLI 直接 `yolo detect train ...` 时，自定义模块未必会自动注册；更稳妥的方式是走当前 Python 入口脚本。
- `custom_blocks.py` 中的消融模块主要用于结构对比，因此请确保训练时加载的 YAML 与论文表格中的模块定义一一对应。
- 如果你修改了自定义模块或新增了消融类，记得同步更新 `register_custom_modules()`。

## References

- PyTorch 官方安装页：<https://pytorch.org/get-started/locally/>
- Ultralytics 安装文档：<https://docs.ultralytics.com/quickstart/>
- Ultralytics Python 用法：<https://docs.ultralytics.com/usage/python/>
- Ultralytics 检测任务说明：<https://docs.ultralytics.com/tasks/detect>
