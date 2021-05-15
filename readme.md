## <b> PMT: Physio Motion Transfer</b> [[Page]](https://fabioo29.github.io/physio-motion-transfer/) [[Paper]](https://arxiv.org/abs/xxxx.xxxx)

<img src="assets/constraints_retargeting.png" width="400"/>  <img src="assets/dataset.gif" width="400" />

## Introduction

This repository contains the original implementation of the Physio Motion Transfer: **"A shape-aware retargeting approach to transfer human motion and appearance in monocular videos for physiotherapy purposes"**. This method uses four main components to get to the final result where the user will see his body animated with movements that they never did.

## Overall

This method is composed by the following **5** main components and **9** DL models:
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
 - Image inpaiting
    1. **BodyDelete** (Remove body from image)
    2. **ImageInpaiting** (full background without the body)  

*NOTE*: Tested on Docker container running on Ubuntu 16.04 LTS.

## Setup and Installation

Go to xyz.com  
Add data(video or frames) for the patient  
Add data(video) with the desired movement  
Patient tries to follow his animated avatar on screen  

## License

Repo licensed with xyz...

## Contacts

xyz@gmail.com // github.com/xyz // linkedin.com/xyz


