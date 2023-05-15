from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class CreateSensor(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.ImageField()
    name = models.CharField(max_length=50, default='default')
    object_name = models.CharField(max_length=50, default='default')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    x = models.IntegerField()
    y = models.IntegerField()
    w = models.IntegerField()
    h = models.IntegerField()

class BlockingArea(models.Model):
    # file = models.ImageField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    x = models.FloatField()
    y = models.FloatField()
    width = models.FloatField()
    height = models.FloatField()