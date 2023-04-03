from django.db import models

# Create your models here.

class Image(models.Model):
    file = models.ImageField()
    name = models.CharField(max_length=50)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    #class Meta:
    #    verbose_name = 'photo'
    #    verbose_name_plural = 'photos'