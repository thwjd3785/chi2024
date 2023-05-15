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

# login 기능
from django.contrib.auth import login
from django.shortcuts import render, redirect
from .forms import LoginForm
from django.contrib.auth.forms import AuthenticationForm

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, request.POST)
        print("login try!!!")
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            print("login try")
            return redirect('camera_prototype_v2:index')
        else:
            print("Form errors:", form.errors)
    else:
        form = AuthenticationForm()
    return render(request, 'camera_prototype_v2/login.html', {'form': form})

from django.contrib.auth import logout
from django.shortcuts import redirect

def user_logout(request):
    logout(request)
    return redirect('camera_prototype_v2:login')

from django.shortcuts import render, redirect
from .forms import UserRegistrationForm

def user_register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            return redirect('camera_prototype_v2:login')
    else:
        form = UserRegistrationForm()
    return render(request, 'camera_prototype_v2/register.html', {'form': form})


###
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('camera_prototype_v2:capture')
            else:
                # Handle case when authentication fails
                pass
    else:
        form = AuthenticationForm()
    return render(request, 'camera_prototype_v2/login.html', {'form': form})

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

webcam_width = 640
webcam_height = 378

# 객체 인식 class 읽어오기
# Load class names
with open('camera_prototype_v2/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

from django.contrib.auth.decorators import login_required
# Create your views here.
@login_required
def home(request):
    # templates의 폴더명(즉 app의 폴더명/___.html)
    return render(request, 'camera_prototype_v2/capture.html')

# type: ignore[attr-defined]
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

font_path = "camera_prototype_v2/NanumGothic-Bold.ttf"

def put_korean_text(image, text, position, font_path, font_size, color):
    font = ImageFont.truetype(font_path, font_size)

    # Reverse the color channels
    reversed_color = (color[2], color[1], color[0])

    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=reversed_color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

kor_to_eng_translations = {
    '사람': 'person',
    '자전거': 'bicycle',
    '자동차': 'car',
    '오토바이': 'motorbike',
    '비행기': 'aeroplane',
    '버스': 'bus',
    '기차': 'train',
    '트럭': 'truck',
    '배': 'boat',
    '신호등': 'traffic light',
    '소화전': 'fire hydrant',
    '정지 표시': 'stop sign',
    '주차권 판매기': 'parking meter',
    '벤치': 'bench',
    '새': 'bird',
    '고앙이': 'cat',
    '개': 'dog',
    '말': 'horse',
    '양': 'sheep',
    '소': 'cow',
    '코끼리': 'elephant',
    '곰': 'bear',
    '얼룩말': 'zebra',
    '기린': 'giraffe',
    '백팩': 'backpack',
    '우산': 'umbrella',
    '핸드백': 'handbag',
    '넥타이': 'tie',
    '여행 가방': 'suitcase',
    '원반': 'frisbee',
    '스키': 'skis',
    '스노보드': 'snowboard',
    '공': 'sports ball',
    '연': 'kite',
    '야구 배트': 'baseball bat',
    '야구 글러브': 'baseball glove',
    '스케이트보드': 'skateboard',
    '서핑보드': 'surfboard',
    '테니스 라켓': 'tennis racket',
    '병': 'bottle',
    '와인잔': 'wine glass',
    '컵': 'cup',
    '포크': 'fork',
    '칼': 'knife',
    '숟가락': 'spoon',
    '그릇': 'bowl',
    '바나나': 'banana',
    '사과': 'apple',
    '샌드위치': 'sandwich',
    '오렌지': 'orange',
    '브로콜리': 'broccoli',
    '당근': 'carrot',
    '핫도그': 'hot dog',
    '피자': 'pizza',
    '도넛': 'donut',
    '케이크': 'cake',
    '의자': 'chair',
    '쇼파': 'sofa',
    '화분': 'pottedplant',
    '침대': 'bed',
    '식탁': 'diningtable',
    '변기': 'toilet',
    '티비': 'tvmonitor',
    '노트북': 'laptop',
    '마우스': 'mouse',
    '리모컨': 'remote',
    '키보드': 'keyboard',
    '핸드폰': 'cell phone',
    '전자레인지': 'microwave',
    '오븐': 'oven',
    '토스터': 'toaster',
    '싱크대': 'sink',
    '냉장고': 'refrigerator',
    '책': 'book',
    '시계': 'clock',
    '꽃병': 'vase',
    '가위': 'scissors',
    '곰인형': 'teddy bear',
    '드라이어': 'hair drier',
    '칫솔': 'toothbrush',
}

def reverseTranslations(translations):
    return {v: k for k, v in translations.items()}

eng_to_kor_translations = reverseTranslations(kor_to_eng_translations)

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

                english_object = classes[int(result[5].item())]
                korean_object = eng_to_kor_translations.get(english_object, english_object)

                label = f"{korean_object}: {round(result[4].item(), 2) * 100}%"

                color = colors[int(result[5].item())]
                int_color = (int(color[0]), int(color[1]), int(color[2]))

                # 박스치고 텍스트 보여주기
                cv2.rectangle(self.frame, (x_min, y_min), (x_max, y_max), int_color, 2)
                # cv2.putText(self.frame, label, (x_min, y_min + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
                self.frame = put_korean_text(self.frame, label, (x_min+10, y_min+10), font_path, 32, int_color)

        if self.sensors:
            for sensor in self.sensors:
                x = int (sensor.x * 1280 / webcam_width)
                y = int (sensor.y * 720 / webcam_height)
                w = int (sensor.w * 1280 / webcam_width)
                h = int (sensor.h * 720 / webcam_height)
                # w = int (sensor.w)
                # h = int (sensor.h)
                sensor_value = self.sensor_values.get(sensor.name, "OFF")  # Get the sensor value, default to "OFF"
                color = (0, 255, 0) if sensor_value == "ON" else (0, 0, 255)  # Green if "ON", Red if "OFF"
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 6)
                # cv2.putText(self.frame, sensor.name, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
                
                self.frame = put_korean_text(self.frame, sensor.name, (x+10, y+10), font_path, 32, color)

        
        if self.blocks:
            for block in self.blocks:
                x = int (block.x * 1280 / webcam_width)
                y = int (block.y * 720 / webcam_height)
                w = int (block.width * 1280 / webcam_width)
                h = int (block.height * 720 / webcam_height)
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
            send_mqtt_message("homeassistant/binary_sensor/sensorBedroom/state", message)

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + jgp_byte_frame + b'\r\n\r\n')

from django.http import HttpResponse

@gzip.gzip_page
def detectme(request):
    try:
        sensors = CreateSensor.objects.filter(user=request.user)
        blocks = BlockingArea.objects.filter(user=request.user)
        cam = VideoCamera(sensors, blocks)
        return StreamingHttpResponse(generate(cam), content_type='multipart/x-mixed-replace;boundary=frame')
    except:
        print("에러입니다...")
        return HttpResponse("Response from detectme")



###
# Import additional modules
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required

@csrf_exempt
def index(request):
    form = CreateSensorForm(request.POST or None, request.FILES or None)
    print("Request Method:", request.method)
    print("Form is valid:", form.is_valid())
    print("Form errors:", form.errors)  # Add this line

    if request.method == 'POST' and form.is_valid():
        image = form.save(commit=False)
        image.user = request.user 
        image.object_name = request.POST.get('object_name')
        image.x = request.POST.get('x')
        image.y = request.POST.get('y')
        image.w = request.POST.get('w')
        image.h = request.POST.get('h')
        image.user = request.user
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

        send_mqtt_message("homeassistant/binary_sensor/sensorBedroom/{}/config".format(filename), messageCreateSensor)

        return JsonResponse({'success': True})
    
    context = {'form':form}
    return render(request, 'camera_prototype_v2/capture.html', context)

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
    x = landmark.x * webcam_width
    y = landmark.y * webcam_height
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
                x_min = x_min * webcam_width / 1280
                x_max = x_max * webcam_width / 1280
                y_min = y_min * webcam_height / 720
                y_max = y_max * webcam_height / 720

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

from django.core import serializers
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required

@login_required
def get_block_data(request):
    if request.user.is_authenticated:
        data = serializers.serialize('json', BlockingArea.objects.filter(user=request.user))
    else:
        data = serializers.serialize('json', [])
    return JsonResponse(data, safe=False)

@login_required
def get_sensor_data(request):
    if request.user.is_authenticated:
        data = serializers.serialize('json', CreateSensor.objects.filter(user=request.user))
    else:
        data = serializers.serialize('json', [])
    return JsonResponse(data, safe=False)


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
            sensor_data = CreateSensor.objects.get(pk=sensor_id, user=request.user)
            past_image_name = sensor_data.name
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

            messageCreateSensor = {
                "device_class": "motion",
                "name": name,
                "object_id": name,
                "unique_id": name,
                "device": {
                    "identifiers": "123",
                    "name": "testDevice",
                    "manufacturer": "VarOfLa",
                    "configuration_url": "https://blog.naver.com/dhksrl0508"
                },
                "state_topic": "homeassistant/binary_sensor/sensorBedroom/state",
                "unit_of_measurement": "%",
                "value_template": "{{ value_json." + name + " }}"
            }

            send_mqtt_message("homeassistant/binary_sensor/sensorBedroom/{}/config".format(past_image_name), messageCreateSensor)


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
        CreateSensor.objects.filter(pk__in=ids, user=request.user).delete()
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

        block.user = request.user

        block.save()

        return JsonResponse({'success': True})
    else:
        return JsonResponse({'success': False})
    
def delete_block_data(request):
    if request.method == 'POST':
        ids = json.loads(request.POST['ids'])
        BlockingArea.objects.filter(pk__in=ids, user=request.user).delete()
        return JsonResponse({"status": "success"})

    return JsonResponse({"status": "error"})