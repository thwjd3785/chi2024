from django.shortcuts import render, get_object_or_404, redirect
from .models import CameraImage # models.py 파일에서 Photo 클래스 불러오기

# POST를 하기 위해 만든 모델링
from .forms import PhotoForm

# 카메라 영상을 갖고 오기 위해서
import cv2 # opencv
import threading # 병렬적으로 하고 싶을때

from django.views.decorators import gzip # 이 데코레이터는 브라우저가 gzip 압축을 허용하는 경우 콘텐츠를 압축합니다.
from django.http import StreamingHttpResponse # 응답을 실시간으로

# 객체 인식을 위한
import torch
import numpy as np

# 객체 인식 class 읽어오기
classes = []
with open('detectme/coco.names', 'r') as f:
    # text 파일 줄 별로 list화
    classes = [line.strip() for line in f.readlines()]

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
colors = np.random.uniform(0,255, size=(len(classes),3))


# Create your views here.
def home(request):
    # templates의 폴더명(즉 app의 폴더명/___.html)
    return render(request, 'detectme/home.html')


# 여기서부터 찐
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        _, self.frame = self.video.read()
        
        # 동시작업을 하기 위해
        threading.Thread(target=self.update, args=()).start()
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        
        # 창에 띄우기 위한 정보
        class_ids = []
        class_probs = []
        boxes = []


        #image = self.frame
        frameRGB = self.frame[..., ::-1] #BGR -> RGB
        results = model(frameRGB)
        for result in results.xyxy[0]:

            if result[4].item() > 0.5:
                x_min = int(result[0].item()) # box의 x 최소값
                x_max = int(result[2].item()) # box의 x 최댓값

                y_min = int(result[1].item()) # box의 y 최소값
                y_max = int(result[3].item()) # box의 y 최댓값

                # Visualization
                # box 너비, 높이 구해서 정보 저장
                w = x_max - x_min
                h = y_max - y_min

                boxes.append([x_min, y_min, w, h])
                class_probs.append(round(result[4].item(),2))  # 클래스 확률 추가
                class_ids.append(int(result[5].item()))  # 클래스 종류 추가

            # 위에서 추가한 box 갯수만큼
        for i in range(len(boxes)):
                    # 박스 정보 다시 풀기
            x, y, w, h = boxes[i]
                    
                    # label 값 보여줄려고
            label = str(classes[class_ids[i]])+": "+str(class_probs[i]*100)+"%"
                    # color도 입히자!
            color = colors[class_ids[i]]
                    
                    # 박스치고 텍스트 보여주기
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(self.frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)

        yolo = self.frame   
        _, jpeg = cv2.imencode('.jpg', yolo)

        # bytes 파일로 바꿔준다.
        return jpeg.tobytes() # bytes 파일로 바꿔준다.
    
    def update(self):
        while True:
            _, self.frame = self.video.read()
            


def generate(camera):
    while True:
        jgp_byte_frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + jgp_byte_frame + b'\r\n\r\n')

@gzip.gzip_page
def detectme(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(generate(cam), content_type = 'multipart/x-mixed-replace;boundary=frame')
    except:
        print("에러입니다...")
        pass

from .forms import PhotoForm


def sensor_list(request):
    # 고급기능
    sensors = CameraImage.objects.all() # Photo model data를 모두 가져오기
    return render(request, 'detectme/sensor_list.html', {'sensors':sensors})

# 세부정보 페이지 열기
def sensor_detail(request, pk): #pk는 데이터베이스의 각 record를 식별하는 기본키(Prmiary Key)의 줄임말
    # Photo로부터 데이터를 찾고 만약에 없으면 404 에러를 반환
    sensor = get_object_or_404(CameraImage, pk=pk)
    return render(request, 'detectme/sensor_detail.html', {'sensor':sensor})


def sensor_post(request):
    if request.method =="POST":
        form = PhotoForm(request.POST)
        if form.is_valid():
            sensor = form.save(commit=False)
            sensor.save()

            # form 이 POST가 완료되면 detail 로 갈거야
            return redirect('sensor_detail', pk=sensor.pk)
    else:
        form = PhotoForm()

    return render(request, 'detectme/sensor_post.html', {'form':form})

# 사진 수정하기
def sensor_edit(request, pk):
    sensor = get_object_or_404(CameraImage, pk=pk)
    if request.method =="POST":
        form = PhotoForm(request.POST, instance=sensor) # 기존 예시를 제공해주자
        if form.is_valid():
            sensor = form.save(commit=False)
            sensor.save()

            # form 이 POST가 완료되면 detail 로 갈거야
            return redirect('sensor_detail', pk=sensor.pk)
    else:
        form = PhotoForm(instance=sensor)

    return render(request, 'detectme/sensor_post.html', {'form':form})

