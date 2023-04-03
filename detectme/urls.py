from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('detectme', views.detectme, name='detectme'),

    path('sensor', views.sensor_list, name='sensor_list'),
    path('sensor/<int:pk>/', views.sensor_detail, name="sensor_detail"),
    # <int:pk> pk라는 이름의 정수형 변수가 들어갈 거다.

    # POST하는 URL
    path('sensor/new/', views.sensor_post, name="sensor_post"),

    # EDIT하는 URL
    path('sensor/<int:pk>/edit', views.sensor_edit, name="sensor_edit"),
]