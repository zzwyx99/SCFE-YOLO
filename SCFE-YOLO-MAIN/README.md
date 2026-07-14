# UAV_yolo12

A UAV small-object detection repository based on Ultralytics YOLO. The current project mainly focuses on `ScaleConsistencyCoupledSAEM` and its ablation variants, with VisDrone as the primary dataset.

Core project files:

- `custom_blocks.py`: custom modules and ablation modules
- `configs/*.yaml`: different model structures and ablation configs
- `trainSCFEYOLO.py`: training entry point, which first registers the custom modules and then hands the model over to Ultralytics
- `tools/manual_flops.py`: manual parameter and FLOPs counter

## 1. Environment

It is recommended to use a dedicated Conda environment.

Recommended new environment:

```bash
conda create -n scfe-yolo python=3.8 -y
conda activate scfe-yolo
```

Install PyTorch. If you use a GPU, please choose the command that matches your local CUDA version from the official PyTorch page:

```bash
pip install torch==2.4.1 torchvision==0.19.1
```

Install Ultralytics and common dependencies:

```bash
pip install ultralytics==8.4.22
pip install opencv-python==4.12.0.88 pandas==2.0.3 scipy==1.10.1 tqdm==4.67.1 matplotlib
```

## 2. Dataset

The repository already contains a `datasets/VisDrone/` directory, including images, labels, and raw archives. The default training setting uses:

```python
model.train(data="VisDrone.yaml", ...)
```

If you do not yet have `VisDrone.yaml` in your environment, you can create one following the Ultralytics detection data format, for example:

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

If your class definitions or label indices are different, please modify `names` according to your actual data.

## 3. Quick Start

You need to register the custom modules before training, otherwise the Ultralytics model parser cannot recognize the custom layer names in `configs/*.yaml`. The current training script already handles this step:

```python
from ultralytics import YOLO
from custom_blocks import register_custom_modules

register_custom_modules()
model = YOLO("configs/NSR.yaml")
```

Run directly:

```bash
python trainSCFEYOLO.py
```

See the default training entry in [trainSCFEYOLO.py](trainSCFEYOLO.py).

If you want to switch to other structures, you can directly modify:

```python
model = YOLO("configs/SCFE-YOLO.yaml")
model = YOLO("configs/NSRBR.yaml")
model = YOLO("configs/NSRBRCC.yaml")
model = YOLO("configs/wobranchrouter.yaml")
model = YOLO("configs/nospatialgate.yaml")
model = YOLO("configs/noFE.yaml")
```

## 4. Configs

The main configs currently included in this repository are:

- `configs/SCFE-YOLO.yaml`
  Full model using `ScaleConsistencyCoupledSAEM`
- `configs/NSR.yaml`
  Keeps NSR only
- `configs/NSRBR.yaml`
  NSR + BR
- `configs/NSRBRCC.yaml`
  NSR + BR + CSP
- `configs/wobranchrouter.yaml`
  Removes the Branch Router
- `configs/nospatialgate.yaml`
  Removes DSG / Spatial Gate
- `configs/noFE.yaml`
  Removes native enhancement and keeps only the cross-scale coupling path
- `configs/nop5.yaml`
  Removes the deepest P5 stage
- `configs/AF-YOLO.yaml`
  Another AF-YOLO-style structure retained in the repository

## 5. Custom Blocks

The custom modules are defined in [custom_blocks.py](custom_blocks.py). The main ones currently included are:

- `ScaleAwareEdgeMixer`
  A basic multi-branch texture/context/edge mixing module
- `ScaleConsistencyCoupledSAEM`
  The full SCC-SAEM, including NSR, BR, CSP, CGA, and DSG
- `ScaleConsistencyCoupledSAEMNSR`
  Keeps only native-scale reference construction
- `ScaleConsistencyCoupledSAEMNSRBR`
  NSR + BR
- `ScaleConsistencyCoupledSAEMNSRBRCSP`
  NSR + BR + CSP
- `ScaleConsistencyCoupledSAEMNSRBRCSPCGA`
  NSR + BR + CSP + CGA
- `ScaleConsistencyCoupledSAEMNoBranchRouter`
  Removes the branch router
- `ScaleConsistencyCoupledSAEMNoSpatialGate`
  Removes the spatial gate
- `ScaleOnlyCoupledSAEM`
  Removes native enhancement and keeps only the scale consistency path

Abbreviation definitions:

- `NSR`: native-scale reference construction
- `BR`: branch router
- `CSP`: compressed-scale perturbation
- `CGA`: consistency-gated alignment
- `DSG`: discrepancy-aware spatial gate

## 6. Notes

- When using the Ultralytics CLI directly, such as `yolo detect train ...`, custom modules may not be registered automatically. Using the current Python entry script is safer.
- The ablation modules in `custom_blocks.py` are mainly used for structural comparison, so please make sure the YAML loaded during training matches the module definitions in your paper table.
- If you modify custom modules or add new ablation classes, remember to update `register_custom_modules()` accordingly.

## References

- PyTorch official installation page: <https://pytorch.org/get-started/locally/>
- Ultralytics installation docs: <https://docs.ultralytics.com/quickstart/>
- Ultralytics Python usage: <https://docs.ultralytics.com/usage/python/>
- Ultralytics detection task docs: <https://docs.ultralytics.com/tasks/detect>
