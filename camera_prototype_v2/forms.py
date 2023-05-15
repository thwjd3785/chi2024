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

from django import forms
from django.contrib.auth.forms import AuthenticationForm

class LoginForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}))

from django import forms
from django.contrib.auth.models import User

class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    password_confirm = forms.CharField(widget=forms.PasswordInput, label='Confirm password')

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        password_confirm = cleaned_data.get('password_confirm')

        if password != password_confirm:
            raise forms.ValidationError("Passwords do not match.")
