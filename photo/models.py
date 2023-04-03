from django.db import models

# models는 Django의 DB와 관련된 내용을 미리 작성해놓은 도구

# Create your models here.
class Photo(models.Model): # models.Model로 상속받아 각 속성들을 models. 으로 정의
    title = models.CharField(max_length=50)   # 문자열
    author = models.CharField(max_length=50)  # 문자열
    image = models.CharField(max_length=200)  # 문자열
    description = models.TextField()          # 문자열(길이제한 필요x)
    # price = models.IntegerField()             # 정수

class CaptureImage(models.Model):
    image = models.ImageField(upload_to="capture-image")
    timestamp = models.DateTimeField(auto_now_add=True)