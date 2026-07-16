# SCFE-YOLO

基于 Ultralytics YOLO 的无人机小目标检测实验仓库，当前项目主要围绕 `ScaleConsistencyCoupledSAEM` 展开，数据集以 VisDrone 为主。

项目核心文件：

- `custom_blocks.py`：自定义模块与消融变体
- `configs/*.yaml`：模型定义与消融配置
- `trainSCFEYOLO.py`：训练入口，负责先注册自定义模块

## 1. Environment

建议使用独立的 Conda 环境。

推荐环境配置：

```bash
conda create -n scfe-yolo python=3.8 -y
conda activate scfe-yolo
```

安装 PyTorch。若使用 GPU，请根据 PyTorch 官方安装页面选择与你本机 CUDA 版本匹配的命令：

```bash
pip install torch==2.4.1 torchvision==0.19.1
```

安装 Ultralytics 和常用依赖：

```bash
pip install ultralytics==8.4.22
pip install opencv-python==4.12.0.88 pandas==2.0.3 scipy==1.10.1 tqdm==4.67.1 matplotlib
```

## 2. Dataset

默认训练脚本使用：

```python
model.train(data="VisDrone.yaml", ...)
```

如果你已经有自己的数据集和对应的数据集 YAML 文件，请直接编辑该 YAML，把数据集根目录路径和类别定义改成你自己的配置。如果你本地还没有准备好 VisDrone，保留 `data="VisDrone.yaml"` 即可，Ultralytics 会在训练时自动下载并准备 VisDrone 数据集。

如果你需要新建或自定义 `VisDrone.yaml`，可以参考 Ultralytics 检测任务的数据格式，例如：

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

如果你的类别定义或标签编号不同，请按自己的数据实际情况修改 `names`。

如果你需要构建尺度扰动数据集，可以使用 `dataset_scalePerturb.py` 基于 YOLO 格式数据集生成高视角扰动版本，并自动同步更新标签。该脚本默认读取 `./datasets/VisDrone`，并将生成后的数据集输出到 `./datasets/` 下。

## 3. Quick Start

训练前需要先注册自定义模块，否则 Ultralytics 的模型解析器无法识别 `configs/*.yaml` 中的自定义层名称。当前训练脚本已经处理了这一步：

```python
from ultralytics import YOLO
from custom_blocks import register_custom_modules

register_custom_modules()
model = YOLO("configs/SCFE-YOLO.yaml")
```

运行：

```bash
python trainSCFEYOLO.py
```

默认训练入口见 [trainSCFEYOLO.py](trainSCFEYOLO.py)。

如果你想切换到其他结构，可以直接修改配置文件路径：

```python
model = YOLO("configs/SCFE-YOLO.yaml")
model = YOLO("configs/AF-YOLO.yaml")
```

## 4. Configs

当前仓库中主要包含以下配置：

- `configs/SCFE-YOLO.yaml`
  完整模型，使用 `ScaleConsistencyCoupledSAEM`
- `configs/AF-YOLO.yaml`
  仓库中保留的另一套 AF-YOLO 风格结构

## 5. Custom Blocks

自定义模块定义在 [custom_blocks.py](custom_blocks.py) 中，目前主要包括：

- `ScaleAwareEdgeMixer`
  基础多分支纹理/上下文/边缘混合模块
- `ScaleConsistencyCoupledSAEM`
  完整版 SCC-SAEM，包含 NSR、BR、CSP、CGA 和 DSG

缩写说明：

- `NSR`：native-scale reference construction
- `BR`：branch router
- `CSP`：compressed-scale perturbation
- `CGA`：consistency-gated alignment
- `DSG`：discrepancy-aware spatial gate

## 6. Notes

- 如果你直接使用 Ultralytics CLI，例如 `yolo detect train ...`，自定义模块可能不会自动注册。当前的 Python 入口脚本更稳妥。
- `custom_blocks.py` 中的消融模块主要用于结构对比，因此请确保训练时加载的 YAML 与论文表格中的模块定义一致。
- 如果你修改了自定义模块或新增了消融类，记得同步更新 `register_custom_modules()`。

## References

- PyTorch 官方安装页：<https://pytorch.org/get-started/locally/>
- Ultralytics 安装文档：<https://docs.ultralytics.com/quickstart/>
- Ultralytics Python 用法：<https://docs.ultralytics.com/usage/python/>
- Ultralytics 检测任务说明：<https://docs.ultralytics.com/tasks/detect>
