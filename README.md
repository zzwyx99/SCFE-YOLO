# SCFE-YOLO

An experimental repository for UAV small-object detection built on Ultralytics YOLO. The current work mainly revolves around `ScaleConsistencyCoupledSAEM`, with VisDrone as the primary dataset.

Core project files:

- `custom_blocks.py`: custom modules and ablation variants
- `configs/*.yaml`: model definitions and ablation configs
- `trainSCFEYOLO.py`: training entry point that registers the custom modules 

## 1. Environment

Using a dedicated Conda environment is recommended.

Recommended setup:

```bash
conda create -n scfe-yolo python=3.8 -y
conda activate scfe-yolo
```

Install PyTorch. If you are using a GPU, choose the command that matches your local CUDA version from the official PyTorch installation page:

```bash
pip install torch==2.4.1 torchvision==0.19.1
```

Install Ultralytics and the common dependencies:

```bash
pip install ultralytics==8.4.22
pip install opencv-python==4.12.0.88 pandas==2.0.3 scipy==1.10.1 tqdm==4.67.1 matplotlib
```

## 2. Dataset

The default training script uses:

```python
model.train(data="VisDrone.yaml", ...)
```

If you already have your own dataset and a dataset YAML file, edit that YAML so it points to the correct dataset root and class definitions for your setup. If you have not prepared VisDrone locally, keeping `data="VisDrone.yaml"` will let Ultralytics automatically download and prepare the VisDrone dataset for training.

If you need to create or customize `VisDrone.yaml`, follow the Ultralytics detection data format, for example:

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

If your class definitions or label indices are different, update `names` to match your own data.

For scale-perturbation dataset generation, you can use `dataset_scalePerturb.py` to build high-view variants from a YOLO-format dataset while updating labels automatically. By default, it reads from `./datasets/VisDrone` and writes the generated datasets under `./datasets/`.

## 3. Quick Start

Before training, you need to register the custom modules. Otherwise, the Ultralytics model parser will not recognize the custom layer names used in `configs/*.yaml`. The current training script already handles this:

```python
from ultralytics import YOLO
from custom_blocks import register_custom_modules

register_custom_modules()
model = YOLO("configs/SCFE-YOLO.yaml")
```

Run:

```bash
python trainSCFEYOLO.py
```

The default training entry is [trainSCFEYOLO.py](trainSCFEYOLO.py).

If you want to switch to a different structure, modify the config path directly:

```python
model = YOLO("configs/SCFE-YOLO.yaml")
model = YOLO("configs/AF-YOLO.yaml")
```

## 4. Configs

The main configs currently included in this repository are:

- `configs/SCFE-YOLO.yaml`
  Full model using `ScaleConsistencyCoupledSAEM`
- `configs/AF-YOLO.yaml`
  Another AF-YOLO-style structure retained in the repository

## 5. Custom Blocks

The custom modules are defined in [custom_blocks.py](custom_blocks.py). The main ones currently included are:

- `ScaleAwareEdgeMixer`
  Base multi-branch texture/context/edge mixing module
- `ScaleConsistencyCoupledSAEM`
  Full SCC-SAEM including NSR, BR, CSP, CGA, and DSG

Abbreviations:

- `NSR`: native-scale reference construction
- `BR`: branch router
- `CSP`: compressed-scale perturbation
- `CGA`: consistency-gated alignment
- `DSG`: discrepancy-aware spatial gate

## 6. Notes

- If you use the Ultralytics CLI directly, such as `yolo detect train ...`, the custom modules may not be registered automatically. The current Python entry script is the safer choice.
- The ablation modules in `custom_blocks.py` are mainly used for structural comparison, so make sure the YAML loaded during training matches the module definitions used in your paper tables.
- If you modify custom modules or add new ablation classes, remember to update `register_custom_modules()` as well.

## References

- PyTorch official installation page: <https://pytorch.org/get-started/locally/>
- Ultralytics installation docs: <https://docs.ultralytics.com/quickstart/>
- Ultralytics Python usage: <https://docs.ultralytics.com/usage/python/>
- Ultralytics detection task docs: <https://docs.ultralytics.com/tasks/detect>
