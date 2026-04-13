# AGENTS.md

## Entry Points
- Main entrypoints are `train.py` and `val.py`; both explicitly `importlib.import_module("models")` and `importlib.import_module("loaders")` to register custom MMDetection3D modules.
- The only checked-in experiment config is `configs/racformer_r50_nuimg_704x256_f8.py`.

## Setup
- This repo expects the older OpenMMLab stack from `README.md`: Python 3.8, PyTorch 2.0.0 + CUDA 11.8, `mmcv-full==1.6.0`, `mmdet==2.28.2`, `mmsegmentation==0.30.0`, `mmdet3d==1.0.0rc6`, `numpy==1.23.5`, `setuptools==59.5.0`.
- Custom CUDA extensions must be compiled from `models/csrc`: `python setup.py build_ext --inplace`.

## Commands
- Train directly with `torchrun --nproc_per_node 8 train.py --config configs/racformer_r50_nuimg_704x256_f8.py`.
- `dist_train.sh` is just a 4-GPU wrapper around that same config.
- Evaluate with `python val.py --config configs/racformer_r50_nuimg_704x256_f8.py --weights checkpoints/racformer_r50_f8.pth` or `torchrun --nproc_per_node 8 val.py ...` for multi-GPU.
- Generate nuScenes sweep metadata with `python tools/gen_sweep_info.py` if the precomputed `nuscenes_infos_*_sweep.pkl` files are missing.

## Data And Path Gotchas
- Dataset paths are hardcoded in more than one place. `configs/racformer_r50_nuimg_704x256_f8.py` uses `dataset_root = "/mnt/diskNvme1/dataset/nuscenes/"`, and `loaders/nuscenes_dataset.py` also instantiates a global `NuScenes(..., dataroot="/mnt/diskNvme1/dataset/nuscenes/")`. If you relocate the dataset, update both.
- The config expects `nuscenes_infos_train_sweep.pkl`, `nuscenes_infos_val_sweep.pkl`, and `nuscenes_infos_test_sweep.pkl` under the dataset root.
- Pretrained backbone weights are expected at `pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth` via `load_from` in the config.

## Repo Layout
- `models/` holds the custom detector (`racformer.py`), head, transformer, and CUDA-backed ops.
- `loaders/` holds custom dataset classes and pipeline registration.
- `tools/` is mostly dataset prep and visualization helpers; `tools/gen_sweep_info.py` is the only setup-critical script.

## Verification Reality
- No repo-local CI workflows, lint config, formatter config, typecheck config, or automated test suite were found. Verification is manual: compile extensions, then run the smallest feasible `val.py` or `train.py` invocation.

## Search Hygiene
- `train.py` copies code into `outputs/<config-stem>/<timestamp>/backup/` on every run. Exclude `outputs/` from searches unless you intentionally want archived snapshots.
- `val.py` hardcodes `jsonfile_prefix='submission_mini'` inside `evaluate()`, so evaluation artifacts go there unless you change code.
