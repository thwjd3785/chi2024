from django.db import models
import cv2

# Create your models here.
class CameraImage(models.Model): # models.Model로 상속받아 각 속성들을 models. 으로 정의
    title = models.CharField(max_length=50)   # 문자열
    author = models.CharField(max_length=50)  # 문자열
    image = models.ImageField(upload_to="camera-image")  # 문자열
    description = models.TextField()          # 문자열(길이제한 필요x)