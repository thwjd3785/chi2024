from PIL import Image
from django import forms
from django.core.files import File
from .models import CreateSensor
from .models import BlockingArea


class CreateSensorForm(forms.ModelForm):
    class Meta:
        model = CreateSensor
        fields = ('file', 'name', 'object_name', 'x', 'y', 'w', 'h')

class CreateBlockingArea(forms.ModelForm):
    class Meta:
        model = BlockingArea
        fields = ('x', 'y', 'width', 'height') # file 없앰