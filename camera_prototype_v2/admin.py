from django.contrib import admin
from .models import CreateSensor
from .models import BlockingArea


# Register your models here.
admin.site.register(CreateSensor)
admin.site.register(BlockingArea)