## <b> PMT: Physio Motion Transfer</b> [[Page]](https://fabioo29.github.io/physio-motion-transfer/) [[Paper]](https://arxiv.org/abs/xxxx.xxxx)

<img src="assets/constraints_retargeting.png" width="400"/>  <img src="assets/dataset.gif" width="400" />

## Introduction

This repository contains the original implementation of Physio Motion Transfer: **"A shape-aware retargeting approach to transfer human motion and appearance in monocular videos for physiotherapy purposes"**. This method uses five main components to get to the final result where the user will see his body animated with movements that he never did.

## Overall

This method is composed by **5** main components and **9** DL models:
 - Body texture extractor
    1. **CIHP_PGN** (Body Instance segmentation)
    2. **OpenPose** (Body Pose segmentation (keypoints))
    3. **Octopus** (Body, CIHP, OpenPose to SMPLfy model vertices)
    4. **Segmantic Human Texture Stitching** (Body, CIHP,Octopus to body texture)
 - Body model extractor
    1. **DensePose** (Body dense pose extractor)
    2. **Tex2Shape** (Body, DensePose to model)  
 - Video movement extractor
    1. **ROMP** (body video movement(1+ frames) to pose extractor)

*NOTE*: Tested on Docker container running Ubuntu 16.04 LTS, Python3.7 and tensorflow-gpu 1.14.

## Setup and Installation
### Dockerhub (fastest option)

```shell
docker pull fabioo29/physio-motion-transfer:pmt
docker run --rm -it physio-motion-transfer
```

### Build image (recommended but slower option)

```shell
docker clone https://github.com/Fabioo29/physio-motion-transfer.git
cd physio-motion-transfer
docker build . -t physio-motion-transfer
docker run --rm -it physio-motion-transfer
```

## Inference  
Go to xyz.com  
Add data(video or frames) for the patient  
Add data(video) with the desired movement  
Start inference

## License

Repo licensed with xyz...

## Contacts

xyz@gmail.com // github.com/xyz // linkedin.com/xyz


