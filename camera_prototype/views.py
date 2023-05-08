import os
import base64
import cv2
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from .models import CreateSensor
from .forms import CreateSensorForm

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
# Load class names
with open('camera_prototype/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Create your views here.
def home(request):
    # templates의 폴더명(즉 app의 폴더명/___.html)
    return render(request, 'camera_prototype/capture.html')

# type: ignore[attr-defined]
import mediapipe as mp

# 여기서부터 찐
class VideoCamera(object):
    def __init__(self, sensors=None, blocks=None):
        self.video = cv2.VideoCapture(0)
        _, self.frame = self.video.read()
        self.lock = Lock()
        self.sensors = sensors or []
        self.sensor_values = {}

        self.blocks = blocks or []

        
        # Rest of the code
        # Initialize MediaPipe Pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        
        # 동시작업을 하기 위해
        threading.Thread(target=self.update, args=()).start()
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):

        with self.lock:  # Ensure thread safety while accessing the frame
            frame = self.frame.copy()

        # Process frame with MediaPipe Pose
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_pose = self.pose.process(frame_rgb)
        results_yolo = model(frame_rgb)
        
        # Draw pose landmarks on the frame
        if result_pose.pose_landmarks:
            self.mp_drawing.draw_landmarks(self.frame, result_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Convert the frame back to BGR
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 창에 띄우기 위한 정보
        class_ids = []
        class_probs = []
        boxes = []

        #image = self.frame
        # frameRGB = self.frame[..., ::-1] #BGR -> RGB
        #results = model(frameRGB)
        for result in results_yolo.xyxy[0]:

            if result[4].item() > 0.5:
                x_min = int(result[0].item()) # box의 x 최소값
                x_max = int(result[2].item()) # box의 x 최댓값

                y_min = int(result[1].item()) # box의 y 최소값
                y_max = int(result[3].item()) # box의 y 최댓값

                # Visualization
                # box 너비, 높이 구해서 정보 저장
                w = x_max - x_min
                h = y_max - y_min

                label = f"{classes[int(result[5].item())]}: {round(result[4].item(), 2) * 100}%"
                color = colors[int(result[5].item())]

                # 박스치고 텍스트 보여주기
                cv2.rectangle(self.frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(self.frame, label, (x_min, y_min + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)

            # 위에서 추가한 box 갯수만큼
        for i in range(len(boxes)):
                    # 박스 정보 다시 풀기
            x, y, w, h = boxes[i]
                    
                    # label 값 보여줄려고
            label = f"{classes[class_ids[i]]}: {class_probs[i]*100}%"
                    # color도 입히자!
            color = colors[class_ids[i]]
                    
                    # 박스치고 텍스트 보여주기
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(self.frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
            
        if self.sensors:
            for sensor in self.sensors:
                x = int (sensor.x * 1280 / 480)
                y = int (sensor.y * 720 / 360)
                w = int (sensor.w * 1280 / 480)
                h = int (sensor.h * 720 / 360)
                # w = int (sensor.w)
                # h = int (sensor.h)
                sensor_value = self.sensor_values.get(sensor.name, "OFF")  # Get the sensor value, default to "OFF"
                color = (0, 255, 0) if sensor_value == "ON" else (0, 0, 255)  # Green if "ON", Red if "OFF"
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 6)
                cv2.putText(self.frame, sensor.name, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        
        if self.blocks:
            for block in self.blocks:
                x = int (block.x * 1280 / 480)
                y = int (block.y * 720 / 360)
                w = int (block.width * 1280 / 480)
                h = int (block.height * 720 / 360)
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0,0,0), -1)
                cv2.putText(self.frame, str(block.pk), (x + 10, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)


    # Rest of the code

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

        sensors = CreateSensor.objects.all()

        frame_rgb = cv2.cvtColor(camera.frame, cv2.COLOR_BGR2RGB)
        result_pose = camera.pose.process(frame_rgb)  # Get the result_pose variable here
        updated_values = process_frame(camera.frame, sensors, model, result_pose)

        for image_name, value in updated_values.items():
            camera.sensor_values[image_name] = value  # Update sensor values in the VideoCamera instance
            message = {
                image_name: value
            }
            ## send_mqtt_message("homeassistant/binary_sensor/sensorBedroom/state", message)

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + jgp_byte_frame + b'\r\n\r\n')

@gzip.gzip_page
def detectme(request):
    try:
        sensors = CreateSensor.objects.all()
        blocks = BlockingArea.objects.all()
        cam = VideoCamera(sensors, blocks)
        return StreamingHttpResponse(generate(cam), content_type='multipart/x-mixed-replace;boundary=frame')
    except:
        print("에러입니다...")
        pass


###
# Import additional modules
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def index(request):
    form = CreateSensorForm(request.POST or None, request.FILES or None)
    print("Request Method:", request.method)
    print("Form is valid:", form.is_valid())
    print("Form errors:", form.errors)  # Add this line

    if request.method == 'POST' and form.is_valid():
        image = form.save(commit=False)
        image.object_name = request.POST.get('object_name')
        image.x = request.POST.get('x')
        image.y = request.POST.get('y')
        image.w = request.POST.get('w')
        image.h = request.POST.get('h')
        print(image.object_name, image.x, image.y, image.w, image.h)
        image.save()

        filename = request.POST.get('name')
        messageCreateSensor = {
            "device_class": "motion",
            "name": filename,
            "object_id": filename,
            "unique_id": filename,
            "device": {
                "identifiers": "123",
                "name": "testDevice",
                "manufacturer": "VarOfLa",
                "configuration_url": "https://blog.naver.com/dhksrl0508"
            },
            "state_topic": "homeassistant/binary_sensor/sensorBedroom/state",
            "unit_of_measurement": "%",
            "value_template": "{{ value_json." + filename + " }}"
        }

        ## send_mqtt_message("homeassistant/binary_sensor/sensorBedroom/{}/config".format(filename), messageCreateSensor)

        return JsonResponse({'success': True})
    
    context = {'form':form}
    return render(request, 'camera_prototype/capture.html', context)

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
    message_string=json.dumps(message)
    print(topic,message_string)
    client.disconnect()

consecutive_detection_counter = {}
consecutive_non_detection_counter = {}

def get_mediapipe_landmark_coordinates(landmark_name, landmarks):
    landmark_index = mp.solutions.pose.PoseLandmark[landmark_name].value
    landmark = landmarks[landmark_index]
    x = landmark.x * 480
    y = landmark.y * 360
    return x, y

def process_frame(frame, sensors, model, result_pose):
    frame_rgb = frame[..., ::-1]  # Convert BGR to RGB
    results_yolo = model(frame_rgb)

    if result_pose.pose_landmarks:
            landmarks = result_pose.pose_landmarks.landmark
    # Initialize a dictionary to store the updated sensor values
    updated_values = {}

    for sensor in sensors:
        area = [sensor.x, sensor.y, sensor.w, sensor.h]
        object_name = sensor.object_name.lower()

        # Initialize the counters for this sensor if they don't exist
        if sensor.name not in consecutive_detection_counter:
            consecutive_detection_counter[sensor.name] = 0
        if sensor.name not in consecutive_non_detection_counter:
            consecutive_non_detection_counter[sensor.name] = 0

        detected = False

        # Check if the specified object is within the cropped area
        for result in results_yolo.xyxy[0]:
            detected_object = classes[int(result[5].item())].lower()
            if detected_object == object_name:
                x_min, y_min, x_max, y_max = result[:4].tolist()
                x_min = x_min * 480 / 1280
                x_max = x_max * 480 / 1280
                y_min = y_min * 360 / 720
                y_max = y_max * 360 / 720

                if (x_min >= area[0] and y_min >= area[1] and x_max <= area[0] + area[2] and y_max <= area[1] + area[3]):
                    consecutive_detection_counter[sensor.name] += 1
                    consecutive_non_detection_counter[sensor.name] = 0
                    detected = True
                    break

        if not detected and result_pose.pose_landmarks and object_name.upper() in mp.solutions.pose.PoseLandmark.__members__:
            x, y = get_mediapipe_landmark_coordinates(object_name.upper(), landmarks)
            if (x >= area[0] and y >= area[1] and x <= area[0] + area[2] and y <= area[1] + area[3]):
                consecutive_detection_counter[sensor.name] += 1
                consecutive_non_detection_counter[sensor.name] = 0
                detected = True

        if not detected:
            consecutive_non_detection_counter[sensor.name] += 1
            consecutive_detection_counter[sensor.name] = 0

        # Check the counters and update the sensor values accordingly
        if consecutive_detection_counter[sensor.name] >= 3:
            updated_values[sensor.name] = "ON"
        elif consecutive_non_detection_counter[sensor.name] >= 3:
            updated_values[sensor.name] = "OFF"

    return updated_values

from django.http import JsonResponse
from django.core import serializers
from .models import CreateSensor

def get_sensor_data(request):
    data = serializers.serialize('json', CreateSensor.objects.all())
    return JsonResponse(data, safe=False, content_type='application/json')

def get_block_data(request):
    data = serializers.serialize('json', BlockingArea.objects.all())
    return JsonResponse(data, safe=False, content_type='application/json')

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import CreateSensor
from django.core.files.storage import default_storage


@csrf_exempt
def update_sensor_data(request):
    if request.method == 'POST':
        sensor_id = request.POST.get('sensor_id')
        name = request.POST.get('name')
        object_name = request.POST.get('object_name')
        image = request.FILES.get('file')
        x = request.POST.get('x')
        y = request.POST.get('y')
        w = request.POST.get('w')
        h = request.POST.get('h')

        try:
            sensor_data = CreateSensor.objects.get(pk=sensor_id)
            sensor_data.name = name
            sensor_data.object_name = object_name

            # Check if a new image is provided and update it
            if image:
                # Delete the old image file
                if sensor_data.file:
                    default_storage.delete(sensor_data.file.path)

                # Save the new image file
                sensor_data.file = image

            sensor_data.x = x
            sensor_data.y = y
            sensor_data.w = w
            sensor_data.h = h

            sensor_data.save()

            return JsonResponse({'status': 'success'})
        except CreateSensor.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Sensor data not found'})
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import CreateSensor

@csrf_exempt
def delete_sensor_data(request):
    if request.method == 'POST':
        ids = json.loads(request.POST['ids'])
        CreateSensor.objects.filter(pk__in=ids).delete()
        return JsonResponse({"status": "success"})

    return JsonResponse({"status": "error"})


from .models import BlockingArea
from .forms import CreateBlockingArea


@csrf_exempt
def save_blocking_area(request):
    if request.method == 'POST':
        form = CreateBlockingArea(request.POST or None, request.FILES or None)
        block = form.save(commit=False)
        block.x = request.POST.get('x')
        block.y = request.POST.get('y')
        block.width = request.POST.get('width')
        block.height = request.POST.get('height')

        block.save()

        return JsonResponse({'success': True})
    else:
        return JsonResponse({'success': False})
    
def delete_block_data(request):
    if request.method == 'POST':
        ids = json.loads(request.POST['ids'])
        BlockingArea.objects.filter(pk__in=ids).delete()
        return JsonResponse({"status": "success"})

    return JsonResponse({"status": "error"})