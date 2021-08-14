## <b> PMT: Physio Motion Transfer</b> [[Page]](https://fabioo29.github.io/physio-motion-transfer/) [[Paper]](https://github.com/fabioo29/physio-motion-transfer)

<img src="assets/constraints_retargeting.png" width="400"/>  <img src="assets/dataset.gif" width="400" />

## Introduction

 <p style='text-align: justify;'> This repository contains the original implementation of Physio Motion Transfer: <b>"A retargeting approach to transfer human motion and appearance in monocular videos for physiotherapy purposes"</b>. This method uses seven deep learning models and can be divided in three parts (Body texture extractor, 3D Body model extractor, Model retargeting). <b>Body texture extractor</b> - we extract the model colors(clothes, body color, hair color) given the body samples to a UV plan textured map.
 <b>3D Body model extractor</b> - we extract the 3D body model firstly using SMPL(naked body model) and then later it gets deformed to match current body sample texture like clothes and hair. <b>Model retargeting</b> - With this last step we extract every frame pose from the movement video and replicate each pose in the final textured 3D body model with color to render the final video.</p>

## Overall

<p align="center">
  <img width="100%" src="assets/diagram.svg">
</p>

This method is composed by **3** main components and **7** DL models:
 - Body texture extractor
    1. **CIHP_PGN** (Body Instance segmentation (Colored body parts))
    2. **OpenPose** (Body Pose estimation (keypoints))
    3. **Octopus** (3D SMPL body naked model)
    4. **Segmantic Human Texture Stitching** (3D mesh texture)
 - 3D Body model extractor
    1. **DensePose** (3D UV map coordinates extractor )
    2. **Tex2Shape** (3D deformated SMPL body model)  
 - Model retargeting
    1. **ROMP** (retarget and render of final posed body model)

*NOTE*: Tested on Ubuntu 16.04 LTS, Python3.7 and tensorflow-gpu 1.14.

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

Preparing virtualenv
```shell
virtualenv venv -p python3.7
source venv/bin/activate
pip install -r requirements.txt
```

Starting web server
```shell
python manage.py migrate
python manage.py collectstatic
python manage.py runserver
```

Uploading Samples 
```
Go to http://localhost:8000/
Upload body pictures (1 or more)
 |_ full body pictures
 |_ first has to be front view
 |_ avoid complex backgrounds
Upload desired movement for the source body
 |_ upload only one video
 |_ make sure the video has only one body
Click start inference and wait for the video to render
```

Understand the UI  

 <p style='text-align: justify;'>Once the video is ready you will be redirected to a new page with (a)the rendered video playing and (b)your main 
webcam. Now the main goal is for you to replicate the (a) pose while checking your live pose with (b).
 The video on (a) also has feedback with colored dots on some joints to help you know what to adjust to reach the desired pose.</p>
 
 ```shell
 joints with red color - cant detect the joint on camera
 joints with orange color - You need to adjust the joint position
 joints with green color - The joint is in the correct position
 ```
 Once both poses ((a) and (b)) match within a 80% rate, the video will play 30 more frames and check your pose again.

## Contacts

fabiodiogo29@gmail.com // github.com/fabioo29 // linkedin.com/in/fabioo29
