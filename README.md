# Kinematic-aware Hierarchical Attention Network for Human Pose Estimation in Videos (WACV 2023)

![The framework of HANet](./docs/assets/HANet.png)

https://openaccess.thecvf.com/content/WACV2023/papers/Jin_Kinematic-Aware_Hierarchical_Attention_Network_for_Human_Pose_Estimation_in_Videos_WACV_2023_paper.pdf

## Contributions

- We propose a novel approach HANet that utilizes the keypoints’ kinematic features, following the laws of physics. Our method addresses temporal issues with these proposed features, effectively mitigates the jitter, and becomes robust to occlusion.

- We propose a hierarchical transformer encoder that incorporates multi-scale spatio-temporal attention. We use multi-scale feature maps, i.e., leverage all layers’ attention maps, and improve performance on benchmarks that provide sparse supervision.

- We propose online mutual learning that enables joint optimization between refined input poses and final poses, which chooses an online learning target by their training losses.

- We conduct extensive experiments on large datasets and demonstrate that our framework improves performance on tasks: 2D pose estimation, 3D pose estimation, body mesh recovery, and sparsely-annotated multi-human 2D pose estimation.

## Getting Started

### Environment Requirement

Clone the repo:

```bash
https://github.com/KyungMinJin/HANet.git
```

Install the HANet requirements using `conda`:

```bash
# conda
conda create env --name HANet python=3.6
conda activate HANet
pip install -r requirements.txt
```

### Prepare Data

Sub-JHMDB data used in our experiment can be downloaded here. Refer to [Official DeciWatch Repository](https://github.com/cure-lab/DeciWatch) for more details about the data arrangement.

[Google Drive](https://drive.google.com/drive/folders/1uLpuRcRbbVqmyndCnuuaW7qRACJaqMX1?usp=sharing)

| Dataset                                  | Pose Estimator                                                               | 3D Pose | 2D Pose | SMPL |
| ---------------------------------------- | ---------------------------------------------------------------------------- | ------- | ------- | ---- |
| [Sub-JHMDB](http://jhmdb.is.tue.mpg.de/) | [SimpleBaseline](https://github.com/microsoft/human-pose-estimation.pytorch) |         | ✔       |      |

### Training

Note that datasets should be downloaded and prepared before training.

Run the commands below to start training on Sub-JHMDB:

```shell script
python train.py --cfg configs/config_jhmdb_simplebaseline_2D.yaml --dataset_name jhmdb --estimator simplebaseline --body_representation 2D
```

### Evaluation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/kinematic-aware-hierarchical-attention/pose-estimation-on-j-hmdb)](https://paperswithcode.com/sota/pose-estimation-on-j-hmdb?p=kinematic-aware-hierarchical-attention)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/kinematic-aware-hierarchical-attention/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=kinematic-aware-hierarchical-attention)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/kinematic-aware-hierarchical-attention/3d-human-pose-estimation-on-aist)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-aist?p=kinematic-aware-hierarchical-attention)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/kinematic-aware-hierarchical-attention/3d-human-pose-estimation-on-human36m)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-human36m?p=kinematic-aware-hierarchical-attention)


**Results on 2D Pose:**

| Dataset   | Estimator      | PCK 0.05 (Input/Output):arrow_up: | PCK 0.1 (Input/Output):arrow_up: | PCK 0.2 (Input/Output):arrow_up: | Checkpoint                                                                                           |
| --------- | -------------- | --------------------------------- | -------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Sub-JHMDB | simplebaseline | 57.3%/91.9%                       | 81.6%/98.3%                      | 93.9%/99.6%                      | [Google Drive](https://drive.google.com/drive/folders/11A5NFkViDgQNyCGGwsmhAbUkwmV36M-E?usp=sharing) |

**Results on 3D Pose:**

| Dataset | Estimator | MPJPE (Input/Output):arrow_down: | Accel (Input/Output):arrow_down: |
| ------- | --------- | ------------------ | ------------------ |
| Human3.6M | FCN | 54.6/52.8                       | 19.2/1.4                     | 
| Human3.6M | Mhformer | 38.3/35.4                  | 0.8/0.8                      | 
| 3DPW | PARE | 78.9/77.1                       | 6.9/6.8                          | 
| AIST++ | SPIN | 107.7/69.2                       | 5.7/5.4                       | 

## Visualization

We prepare all visualization codes as soon as possible.

### 2D Pose

Visualize comparison on Sub-JHMDB

![visualize of Sub-JHMDB 2D SimpleBaseline](./docs/assets/jhmdb.gif)

### 3D Pose

Visualize comparison on AIST++

![visualize of AIST++ 3D SPIN](./docs/assets/aist_3D.gif)

### 3D Body Mesh Recovery

Visualize comparison on 3DPW

![visualize of AIST++ SMPL SPIN](./docs/assets/pw3d_smpl.gif)

Visualize comparison on AIST++

![visualize of AIST++ SMPL SPIN](./docs/assets/aist_smpl.gif)

## Citation

```
@inproceedings{jin2023kinematic,
  title={Kinematic-aware Hierarchical Attention Network for Human Pose Estimation in Videos},
  author={Jin, Kyung-Min and Lim, Byoung-Sung and Lee, Gun-Hee and Kang, Tae-Kyung and Lee, Seong-Whan},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5725--5734},
  year={2023}
}
```

## Acknowledgement

- The code is based on [Deciwatch](https://github.com/cure-lab/DeciWatch). Thanks for their well-organized code!
