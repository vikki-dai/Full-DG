# Overcoming Data Deficiency for Multi-person Pose Estimation
# Introduction
Building multi-person pose estimation (MPPE) models that can handle complex foreground and uncommon scenes is an important challenge in computer vision. Aside from designing novel models, strengthening training data is a promising direction but remains largely unexploited for the MPPE task. In this paper, we systematically identify the key deficiencies of existing pose datasets that prevent the power of well-designed models from being fully exploited, and propose corresponding solutions. Specifically, we find that traditional data augmentation techniques are inadequate in addressing the two key deficiencies, imbalanced instance complexity (evaluated by our new metric Instance Complexity) and insufficient realistic scenes. To overcome these deficiencies, we propose a model-agnostic **Full-view Data Generation (Full-DG)** method to enrich the training data from the perspectives of both poses and scenes. By hallucinating images with more balanced pose complexity and richer real-world scenes, Full-DG can help improve pose estimators' robustness and generalizability. In addition, we introduce a plug-and-play **Adaptive Category-aware Loss (AC-Loss)** to alleviate the severe pixel-level imbalance between keypoints and backgrounds (i.e.\ around 1:600). Full-DG together with AC-Loss can be readily applied to both bottom-up and top-down approaches to improve their accuracy. Notably, plugging into the representative estimators HigherHRNet and HRNet, our method achieves substantial performance gains of 1.0%-2.9% AP on COCO benchmark, and 1.0%-5.1% AP on CrowdPose benchmark.

![](https://github.com/vikki-dai/RSGNet/blob/main/figures/framework_RSGNet.png)
# Main Results
## Results on CrowdPose test dataset
![](https://github.com/vikki-dai/RSGNet/blob/main/visualization/main_results_CrowdPose.png)
**Note**:
1. Flip test is used.
2. Person detector has person AP of 71.0 on CrowdPose test dataset.
3. GFLOPs is for convolution and linear layers only.
## Results on  COCO val2017 dataset
![](https://github.com/vikki-dai/RSGNet/blob/main/visualization/main_results_COCOval.png)
**Note**:
1. Flip test is used.
2. Person detector has person AP of 56.4 on COCO val2017 dataset.
3. GFLOPs is for convolution and linear layers only.
## Results on COCO test-dev2017 dataset
![](https://github.com/vikki-dai/RSGNet/blob/main/visualization/main_results_COCO_testdev.png)
**Note**:
1. Flip test is used.
2. Person detector has person AP of 60.9 on COCO test-dev2017 dataset.
3. GFLOPs is for convolution and linear layers only.
# Environment
The code is developed based on the [HRNet project](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch). NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA RTX GPU cards. Other platforms or GPU cards are not fully tested.
# Installation
1. Install pytorch >= v1.0.0 following official instruction. Note that if you use pytorch's version < v1.0.0, you should following the instruction at https://github.com/Microsoft/human-pose-estimation.pytorch to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install requirments：
```python
  pip install -r requirements.txt
```
4. Make libs:
```python
  cd ${POSE_ROOT}/lib
  make
```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
```python
  # COCOAPI=/path/to/clone/cocoapi
  git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
  cd $COCOAPI/PythonAPI
  # Install into global site-packages
  make install
  # Alternatively, if you do not have permissions or prefer
  # not to install the COCO API into global site-packages
  python3 setup.py install --user 
```
6. Install [CrowdPoseAPI](https://github.com/Jeff-sjtu/CrowdPose)
```python
  Install CrowdPoseAPI exactly the same as COCOAPI.
  Reverse the bug stated in https://github.com/Jeff-sjtu/CrowdPose/commit/785e70d269a554b2ba29daf137354103221f479e**
```
7. Init output and log directory:
```python
  mkdir output 
  mkdir log
```
# Data Preparation
* For **COCO data**, please download from [COCO download](https://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download and extract them under {POSE_ROOT}/data.  

* For **CrowdPose data**, please download from [CrowdPose download](https://github.com/Jeff-sjtu/CrowdPose#dataset), Train/Val is needed for CrowdPose keypoints training and validation. Please download and extract them under {POSE_ROOT}/data.
# Training and Testing
* Testing on CrowdPose dataset using [model zoo's models](https://github.com/vikki-dai/RSGNet/blob/main/model_zoo.txt)
```python
  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/cp_test.py \
  --cfg experiments/crowdpose/hrnet/rsgnet_w32_256x192_adam_lr1e-3.yaml \
  TEST.MODEL_FILE cp_rsgnet_w32_256.pth
```
* Training on CrowdPose dataset
```python
  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/cp_train.py \
  --cfg experiments/crowdpose/hrnet/rsgnet_w32_256x192_adam_lr1e-3.yaml \
```
* Testing on COCO-val dataset using [model zoo's models](https://github.com/vikki-dai/RSGNet/blob/main/model_zoo.txt)
```python
  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/rsgnet_test.py \
  --cfg experiments/cocoe/hrnet/rsgnet_w32_256x192_adam_lr1e-3.yaml \
  TEST.MODEL_FILE coco_rsgnet_w32_256.pth
```
* Training on COCO-val dataset
```python
  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/rsgnet_train.py \
  --cfg experiments/coco/hrnet/rsgnet_w32_256x192_adam_lr1e-3.yaml \
```
