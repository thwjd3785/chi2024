from django.contrib import admin

# Register your models here.
from .models import CameraImage # models.py 파일에서 Photo 클래스 불러오기

# 만든 Photo 모델을 admin 페이지에서 관리할 수 있게 등록
admin.site.register(CameraImage)