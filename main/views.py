import docker
import os
import subprocess
import shutil
import cv2
import numpy as np
import pickle as pkl
import time
import mediapipe as mp
mp_pose = mp.solutions.pose

from PIL import Image

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, HttpResponseForbidden
from django.http import StreamingHttpResponse
from django.views.decorators.http import condition

from .models import ImagesUp, VideoUp

[obj.delete() for obj in ImagesUp.objects.all()]
[obj.delete() for obj in VideoUp.objects.all()]

myDict = {
    'statusImage': None, 
    'statusVideo': None, 
    'Images': None, 
    'Video': None, 
    'task_n': 0, 
    'task_desc': 'Starting up...',
    'percentage': '0',
    'percLevel': 'warning'
    }

def getImg(client):
    for img in client.images.list():
        for tag in img.tags:
            if 'pmt' in tag:
                return img
    
    return False
    
client = docker.from_env()
myImg = getImg(client)

if not myImg:
    try:
        myImg = client.images.pull('fabioo29/physio-motion-transfer:pmt', tag='pmt')
    except: 
        myImg = client.images.build(path='src', tag='pmt')

myContainer = [cnt for cnt in client.containers.list(all=True) if myImg.short_id.split(':')[-1] in cnt.image.id]    

for cnt in myContainer:
    cnt.remove(v=True, force=True)

volumes=[                                                                                                                                                         
    'type=bind,source={},target={}'.format(os.getcwd() + '/docker/assets', '/root/pmt/assets'),
    'type=bind,source={},target={}'.format(os.getcwd() + '/docker/output', '/root/pmt/output'),
    'type=bind,source={},target={}'.format(os.getcwd() + '/docker/body_samples', '/root/pmt/body_samples'),
    'type=bind,source={},target={}'.format(os.getcwd() + '/docker/move_samples', '/root/pmt/move_samples'),
    'type=bind,source={},target={}'.format(os.getcwd() + '/docker/assets_decoded', '/root/pmt/assets_decoded')
]

def get_angle(p0, p1=np.array([0,0]), p2=None):
    ''' compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    '''
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def comparePose(img1, img2):

    with mp_pose.Pose(static_image_mode=True) as pose:
        metrics = []
        poseAux = []
        keypointAux = []

        for image in [img1, img2]:
            try:
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
                if not results.pose_landmarks:
                    return False
            except:
                return False

            poseMetrics = []
            saveKeypoints = []

            poseMetrics.append(abs(get_angle(
                [results.pose_landmarks.landmark[16].x * 1080, results.pose_landmarks.landmark[16].y * 1080], 
                [results.pose_landmarks.landmark[14].x * 1080, results.pose_landmarks.landmark[14].y * 1080], 
                [results.pose_landmarks.landmark[12].x * 1080, results.pose_landmarks.landmark[12].y * 1080])))
            saveKeypoints.append([
                results.pose_landmarks.landmark[16], 
                results.pose_landmarks.landmark[14], 
                results.pose_landmarks.landmark[12]
            ])
            poseMetrics.append(abs(get_angle(
                [results.pose_landmarks.landmark[14].x * 1080, results.pose_landmarks.landmark[14].y * 1080], 
                [results.pose_landmarks.landmark[12].x * 1080, results.pose_landmarks.landmark[12].y * 1080], 
                [results.pose_landmarks.landmark[24].x * 1080, results.pose_landmarks.landmark[24].y * 1080])))
            saveKeypoints.append([
                results.pose_landmarks.landmark[14], 
                results.pose_landmarks.landmark[12], 
                results.pose_landmarks.landmark[12]
            ])
            poseMetrics.append(abs(get_angle(
                [results.pose_landmarks.landmark[13].x * 1080, results.pose_landmarks.landmark[13].y * 1080], 
                [results.pose_landmarks.landmark[11].x * 1080, results.pose_landmarks.landmark[11].y * 1080], 
                [results.pose_landmarks.landmark[23].x * 1080, results.pose_landmarks.landmark[23].y * 1080])))
            saveKeypoints.append([
                results.pose_landmarks.landmark[13], 
                results.pose_landmarks.landmark[11], 
                results.pose_landmarks.landmark[23]
            ])
            poseMetrics.append(abs(get_angle(
                [results.pose_landmarks.landmark[11].x * 1080, results.pose_landmarks.landmark[11].y * 1080], 
                [results.pose_landmarks.landmark[13].x * 1080, results.pose_landmarks.landmark[13].y * 1080], 
                [results.pose_landmarks.landmark[15].x * 1080, results.pose_landmarks.landmark[15].y * 1080])))
            saveKeypoints.append([
                results.pose_landmarks.landmark[11], 
                results.pose_landmarks.landmark[13], 
                results.pose_landmarks.landmark[15]
            ])
            poseMetrics.append(abs(get_angle(
                [results.pose_landmarks.landmark[28].x * 1080, results.pose_landmarks.landmark[28].y * 1080], 
                [results.pose_landmarks.landmark[26].x * 1080, results.pose_landmarks.landmark[26].y * 1080], 
                [results.pose_landmarks.landmark[24].x * 1080, results.pose_landmarks.landmark[24].y * 1080])))
            saveKeypoints.append([
                results.pose_landmarks.landmark[28], 
                results.pose_landmarks.landmark[26], 
                results.pose_landmarks.landmark[26]
            ])
            poseMetrics.append(abs(get_angle(
                [results.pose_landmarks.landmark[27].x * 1080, results.pose_landmarks.landmark[27].y * 1080], 
                [results.pose_landmarks.landmark[25].x * 1080, results.pose_landmarks.landmark[25].y * 1080], 
                [results.pose_landmarks.landmark[23].x * 1080, results.pose_landmarks.landmark[23].y * 1080])))
            saveKeypoints.append([
                results.pose_landmarks.landmark[27], 
                results.pose_landmarks.landmark[25], 
                results.pose_landmarks.landmark[23]
            ])
            poseMetrics.append(abs(get_angle(
                [results.pose_landmarks.landmark[26].x * 1080, results.pose_landmarks.landmark[26].y * 1080], 
                [results.pose_landmarks.landmark[23].x * 1080, results.pose_landmarks.landmark[23].y * 1080], 
                [results.pose_landmarks.landmark[25].x * 1080, results.pose_landmarks.landmark[25].y * 1080])))
            results.pose_landmarks.landmark[23].x -= (results.pose_landmarks.landmark[23].x - results.pose_landmarks.landmark[24].x)/2
            saveKeypoints.append([
                results.pose_landmarks.landmark[26], 
                results.pose_landmarks.landmark[23], 
                results.pose_landmarks.landmark[25]
            ])

            poseAux.append(poseMetrics)
            keypointAux.append(saveKeypoints)

        metrics = sum([m1/m2 if m1 < m2 else m2/m1 for m1,m2 in zip(poseAux[0], poseAux[1])]) / len(poseAux[0])
        poseK1, poseK2 = keypointAux
        poseM1, poseM2 = poseAux

        for kps1, kps2, pm1, pm2 in zip(poseK1, poseK2, poseM1, poseM2):
            for checkk1, checkk2 in zip(kps1, kps2):
                if checkk2.visibility < 0.9:
                    metrics = 0.2
                    img1 = cv2.circle(img1, (int(checkk1.x * 1080), int(checkk1.y * 1080)), 15, (0, 0, 255), -1)
                elif (pm1/pm2 if pm1 < pm2 else pm2/pm1) > 0.8:
                    img1 = cv2.circle(img1, (int(kps1[1].x * 1080), int(kps1[1].y * 1080)), 15, (0, 255, 0), -1)
                else:
                    img1 = cv2.circle(img1, (int(kps1[1].x * 1080), int(kps1[1].y * 1080)), 15, (0, 165, 255), -1)
        
        cv2.imwrite('docker/compare_skeletons/vid_frame.jpg', img1)
        pkl.dump(metrics, open('docker/assets/pose_metrics.txt', 'wb'))

subprocess.call('docker run -itd --gpus=all --mount {} --mount {} --mount {} --mount {} --mount {} -w="/root/pmt" {} bash'.format(volumes[0], volumes[1], volumes[2], volumes[3], volumes[4], myImg.short_id.split(':')[-1]), shell=True)
myContainer = [cnt for cnt in client.containers.list(all=True) if myImg.short_id.split(':')[-1] in cnt.image.id][0]
myContainer.stop()

class mycamera(object):

    def __init__(self):
        self.frames = cv2.VideoCapture(0)
        self.timestamp = time.time()

    def __del__(self):
        self.frames.release()

    def get_jpg_frame(self):
        is_captured, frame = self.frames.read()

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        canva = Image.new('RGB', (1080, 1080), color = 'white')
        img_w, img_h = pil_frame.size[:2]

        if not(img_w == 1080 and img_h == 1080):
            if img_w > img_h:
                resize_rate = img_w / 1080
            else:
                resize_rate = img_h / 1080
            pil_frame = pil_frame.resize((int(img_w/resize_rate), int(img_h/resize_rate)))
            img_w, img_h = pil_frame.size[:2]

        canva.paste(pil_frame, ((int(1080/2-img_w/2)), (int(1080/2-img_h/2))))
        canva = cv2.cvtColor(np.asarray(canva), cv2.COLOR_BGR2RGB)

        frame = cv2.putText(
            cv2.flip(frame, 1), 
            str(int(1.00/(time.time() - self.timestamp))), 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0,255,255), 2, cv2.LINE_AA)

        cv2.imwrite('docker/compare_skeletons/cam_frame.jpg', cv2.flip(canva, 1))
        self.timestamp = time.time()

        retval, jframe = cv2.imencode('.jpg', frame)
        return frame, jframe.tobytes()

def livefeed():
    if 'camera_object' not in locals():
        camera_object = mycamera()
    while True:
        _ , jframe_bytes = camera_object.get_jpg_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jframe_bytes + b'\r\n\r\n')

@condition(etag_func=None)
def livefe(self):
     return  StreamingHttpResponse(
            livefeed(),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )

class outputVideo(object):

    def __init__(self):
        self.cnt = 0
        self.batch = 20
        self.timestamp = time.time()

    def get_jpg_frame(self):
        try:
            if self.cnt != self.batch:
                frame_vid = cv2.flip(cv2.imread('docker/output/{}.jpg'.format(self.cnt)), 1)
                
                if frame_vid is None:
                    # self.cnt = 0
                    frame_vid = cv2.flip(cv2.GaussianBlur(frame_vid,(5,5),cv2.BORDER_DEFAULT), 1)
                
                cv2.imwrite('docker/compare_skeletons/vid_frame.jpg', frame_vid)
                
                myDict['percLevel'] = 'grey'
                myDict['percentage'] = '%'

                if (time.time() - self.timestamp > 1/15):
                    self.cnt = self.cnt + 1
                    self.timestamp = time.time()

            elif self.cnt == self.batch:
                frame_vid = cv2.flip(cv2.imread('docker/compare_skeletons/vid_frame.jpg'), 1)

                if os.path.exists('docker/assets/pose_metrics.txt'):
                    myDict['percentage'] = int(float(pkl.load(open('docker/assets/pose_metrics.txt', 'rb'))*100))
                    if myDict['percentage'] < 30: myDict['percLevel'] = 'red'
                    elif myDict['percentage'] > 80:  
                        os.remove('docker/assets/pose_metrics.txt')
                        myDict['percLevel'] = 'green'
                        self.batch += 20
                        retval, jframe = cv2.imencode('.jpg', frame_vid)
                        return frame_vid, jframe.tobytes()

                    else: myDict['percLevel'] = 'orange'

            if time.time() - self.timestamp >= 0.5:
                comparePose(cv2.imread('docker/output/{}.jpg'.format(self.cnt)), cv2.imread('docker/compare_skeletons/cam_frame.jpg'))
                self.timestamp = time.time()
        except:
            frame_vid = np.ones(shape=(1080, 1080))

        retval, jframe = cv2.imencode('.jpg', frame_vid)
        return frame_vid, jframe.tobytes()

def videofeed():
    if 'video_object' not in locals():
        video_object = outputVideo()
    while True:
        _ , jframe_bytes = video_object.get_jpg_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jframe_bytes + b'\r\n\r\n')

@condition(etag_func=None)
def videofe(self):
     return  StreamingHttpResponse(
            videofeed(),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )

def index(request):
    if not os.path.exists('docker'): os.mkdir('docker')
    if not os.path.exists('docker/body_samples'): os.mkdir('docker/body_samples')
    if not os.path.exists('docker/move_samples'): os.mkdir('docker/move_samples')
    if not os.path.exists('docker/output'): os.mkdir('docker/output')
    if not os.path.exists('docker/assets'): os.mkdir('docker/assets')
    if not os.path.exists('docker/assets_decoded'): os.mkdir('docker/assets_decoded')
    if not os.path.exists('docker/compare_skeletons'): os.mkdir('docker/compare_skeletons')
    return render(request, "index.html", myDict)

def upload(request):
    if request.method == 'POST' and request.FILES:
        if 'image' in request.FILES.keys():
            myimage = request.FILES['image']

            statusimage = 'dup'
            if not ImagesUp.objects.filter(name=myimage.name).count():
                statusimage = 'ext'
                if  str(myimage.name).endswith('.png') or str(myimage.name).endswith('.jpg'):
                    statusimage = None
                    ImagesUp(name=myimage.name, image=myimage).save()
            
            myDict['statusImage'] = statusimage

        if 'video' in request.FILES.keys():
            myvideo = request.FILES['video']

            statusvideo = 'ext'
            if str(myvideo.name).endswith('.mp4'):
                statusvideo = 'max'
                if  not VideoUp.objects.values_list('name').all().count():
                    statusvideo = None
                    VideoUp(name=myvideo.name, video=myvideo).save()

            myDict['statusVideo'] = statusvideo

        myDict['Images'] = ['/assets/' + str(f[0]) for f in ImagesUp.objects.values_list('name').all()]
        myDict['Video'] = ['/assets/' + str(f[0]) for f in VideoUp.objects.values_list('name').all()]

        return render(request, "index.html", myDict)
    
    myDict['statusImage'] = None
    myDict['Images'] = ['/assets/' + str(f[0]) for f in ImagesUp.objects.values_list('name').all()]

    myDict['statusVideo'] = None
    myDict['Video'] = ['/assets/' + str(f[0]) for f in VideoUp.objects.values_list('name').all()]

    return render(request, "index.html", myDict)

def clear_main(request):
    [obj.delete() for obj in ImagesUp.objects.all()]
    [obj.delete() for obj in VideoUp.objects.all()]
    myDict = {'statusImage': None, 'statusVideo': None, 'Images': None, 'Video': None}
    return render(request, "index.html", myDict)

def loadingPage(request):
    myDict['movePic'] = None
    myDict['camPic'] = None
    return render(request, "loading.html", myDict)

def getVideo(request):

    for path in os.listdir('docker'):
        shutil.rmtree('docker/' + path, ignore_errors=True)
        os.makedirs('docker/' + path)

    for file in os.listdir('static/assets/'):
        if '.mp4' in file:
            vidcap = cv2.VideoCapture('static/assets/' + file)

            try:
                fps = vidcap.get(cv2.CAP_PROP_FPS)
            except:
                fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)

            success,image = vidcap.read()
            
            frames = []
            while success:
                frames.append(image)       
                success,image = vidcap.read()

            frames = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in frames]

            for cnt, image in enumerate(frames):
                canva = Image.new('RGB', (1080, 1080), color = 'white')
                img_w, img_h = image.size[:2]

                if not(img_w == 1080 and img_h == 1080):
                    if img_w > img_h:
                        resize_rate = img_w / 1080
                    else:
                        resize_rate = img_h / 1080
                    image = image.resize((int(img_w/resize_rate), int(img_h/resize_rate)))
                    img_w, img_h = image.size[:2]

                canva.paste(image, ((int(1080/2-img_w/2)), (int(1080/2-img_h/2))))
                frames[cnt] = canva

            videodims = (1080,1080)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter('docker/move_samples/' + file,fourcc,fps,videodims)

            frames = [cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR) for img in frames]

            for img in frames:
                video.write(img)
            video.release()
        else:
            canva = Image.new('RGB', (1080, 1080), color = 'white')
            image = Image.open('static/assets/' + file)
            img_w, img_h = image.size[:2]

            if not(img_w == 1080 and img_h == 1080):
                if img_w > img_h:
                    resize_rate = img_w / 1080
                else:
                    resize_rate = img_h / 1080
                image = image.resize((int(img_w/resize_rate), int(img_h/resize_rate)))
                img_w, img_h = image.size[:2]

            canva.paste(image, ((int(1080/2-img_w/2)), (int(1080/2-img_h/2))))
            canva.save('docker/body_samples/' + file)
    
    myContainer.stop()
    myDict['task_n'] = 1
    myDict['task_desc'] = 'Predicting body segmentation from samples...'
    myContainer.start()
    myContainer.exec_run('python infer.py -s1')
    myContainer.stop()
    myDict['task_n'] = 2
    myDict['task_desc'] = 'Predicting SMPL Body model params...'
    myContainer.start()
    myContainer.exec_run('python infer.py -s2')
    myContainer.stop()
    myDict['task_n'] = 3
    myDict['task_desc'] = 'Predicting model texture from samples...'
    myContainer.start()
    myContainer.exec_run('python infer.py -s3')
    myContainer.stop()
    myDict['task_n'] = 4
    myDict['task_desc'] = 'Predicting densePose data from samples...'
    myContainer.start()
    myContainer.exec_run('python infer.py -s4')
    myContainer.stop()
    myDict['task_n'] = 5
    myDict['task_desc'] = 'Predicting pose data from video frames...'
    myContainer.start()
    myContainer.exec_run('python infer.py -s5')
    myContainer.stop()
    myDict['task_n'] = 6
    myDict['task_desc'] = 'Predicting SMPL Body model deformation based on samples data...'
    myContainer.start()
    myContainer.exec_run('python infer.py -s6')
    myContainer.stop()
    myDict['task_n'] = 7
    myDict['task_desc'] = 'Merging model shape with texture for all frames...'
    myContainer.start()
    myContainer.exec_run('python infer.py -s7')
    myContainer.stop()
    myDict['task_n'] = 8
    myDict['task_desc'] = 'Rendering final video frames...'
    myContainer.start()
    myContainer.exec_run('python infer.py -s8')
    myContainer.stop()
    
    shutil.rmtree('assets', ignore_errors=True)

    return render(request, "ui.html", myDict)

def playVideo(request):
    return render(request, 'ui.html', myDict)