from django import forms
from .models import CameraImage

# POST를 하기 위해 억지로 만든 메타데이터
class PhotoForm(forms.ModelForm):
    # data 구조가 이렇게 생겼습니다라고 해주는 data
    class Meta:
        model = CameraImage
        fields = (
            'title',
            'author',
            'image',
            'description',
        )