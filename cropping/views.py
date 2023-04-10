import os
import base64
import cv2
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from .models import Image
from .forms import ImageForm

import paho.mqtt.client as mqtt
from random import randrange, uniform
import time
import json

###
from django.shortcuts import render, get_object_or_404, redirect
# 카메라 영상을 갖고 오기 위해서
import cv2 # opencv
import threading # 병렬적으로 하고 싶을때

from django.views.decorators import gzip # 이 데코레이터는 브라우저가 gzip 압축을 허용하는 경우 콘텐츠를 압축합니다.
from django.http import StreamingHttpResponse # 응답을 실시간으로

# 객체 인식을 위한
import torch
import numpy as np
from threading import Lock

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
        self.lock = Lock()
        
        # 동시작업을 하기 위해
        threading.Thread(target=self.update, args=()).start()
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):

        with self.lock:  # Ensure thread safety while accessing the frame
            frame = self.frame.copy()

        # Resize the frame to a smaller size
        frame = cv2.resize(frame, (480, 360))
        
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
            _, frame = self.video.read()
            with self.lock:  # Ensure thread safety while updating the frame
                self.frame = frame
            


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

###
# Import additional modules
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Add object_detection view function
@csrf_exempt
def object_detection(request):
    if request.method == 'POST':
        try:
            detection_data = json.loads(request.body)

            object_name = detection_data['objectName']
            x = int(detection_data['x'])
            y = int(detection_data['y'])
            width = int(detection_data['width'])
            height = int(detection_data['height'])

            # Process the video frame and detect the specified object within the cropped area
            with VideoCamera().lock:
                frame = VideoCamera().frame.copy()

            cropped_frame = frame[y:y+height, x:x+width]

            frameRGB = cropped_frame[..., ::-1]  # BGR -> RGB
            results = model(frameRGB)

            result = False
            for detection in results.xyxy[0]:
                if detection[4].item() > 0.5:
                    detected_object_name = classes[int(detection[5].item())]
                    if detected_object_name == object_name:
                        result = True
                        break

            return JsonResponse({'result': result})
        except Exception as e:
            print(e)
            return JsonResponse({'result': False}, status=400)

    return JsonResponse({'result': False}, status=405)  # Method not allowed

###

@csrf_exempt
def index(request):
    form = ImageForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()
        filename = request.POST.get('name')
        messageCreateSensor =  {
            "device_class": "motion",
            "name": "{}".format(filename),
            "object_id": "{}".format(filename),
            "unique_id": "{}".format(filename),
            "device": {
                "identifiers": "123",
                "name": "testDevice",
                "manufacturer": "VarOfLa",
                "configuration_url": "https://blog.naver.com/dhksrl0508"
            },
            "state_topic": "homeassistant/binary_sensor/sensorBedroom/{}/state".format(filename),
            "unit_of_measurement": "%",
            "value_template": "{{'{{ value_json.{}}}'}}".format(filename)
        }

        send_mqtt_message("homeassistant/binary_sensor/sensorBedroom/{}/config".format(filename), messageCreateSensor)

        return JsonResponse({'success': True})
    
    context = {'form':form}
    return render(request, 'cropping/capture.html', context)

def send_mqtt_message(topic, message):
    # client = mqtt.Client()
    # client.connect("mqtt.eclipse.org", 1883, 60)
    # client.publish(topic, message)
    # client.disconnect()

    hostname = 'homeassistant.local'
    port = 1883
    timeout = 60

    # Client name을 지정하기
    client = mqtt.Client("HELLO")
    client.username_pw_set("mqtt", "1234")
    client.connect(hostname,port,timeout)
    client.publish(topic, payload=json.dumps(message), retain=False)
    client.disconnect()

'''
@csrf_exempt
def save_image(request):
    if request.method == 'POST':
        # Decode the base64-encoded image data
        dataUrl = request.POST.get('image')
        try:
            _, imgstr = dataUrl.split(';base64,')
        except ValueError:
            return JsonResponse({'success': False, 'message': 'Invalid image data'})
        image_data = base64.b64decode(imgstr)
        # Decode the image data to an OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Crop the image to the desired size
        x = int(request.POST.get('x'))
        y = int(request.POST.get('y'))
        w = int(request.POST.get('w'))
        h = int(request.POST.get('h'))
        img_cropped = img[y:y+h, x:x+w]
        # Create a filename for the image
        dir_path = os.path.join('capture-image')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        files = os.listdir(dir_path)
        files_count = len(files)
        filename = request.POST.get('filename')
        # filename = os.path.join(dir_path, f'image_{files_count}.png')
        # Save the image to the server
        cv2.imwrite(filename, img_cropped)
    
        return JsonResponse({'success': True})
    else:
        return JsonResponse({'success': False, 'message': 'Invalid request method'})
'''
        