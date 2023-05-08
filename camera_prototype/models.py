from django.db import models

# Create your models here.

class CreateSensor(models.Model):
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
    x = models.FloatField()
    y = models.FloatField()
    width = models.FloatField()
    height = models.FloatField()