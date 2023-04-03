from django.urls import path
from . import views

urlpatterns = [
    path('', views.photo_list, name='photo_list'),
    path('photo/<int:pk>/', views.photo_detail, name="photo_detail"),
    # <int:pk> pk라는 이름의 정수형 변수가 들어갈 거다.

    # POST하는 URL
    path('photo/new/', views.photo_post, name="photo_post"),

    # EDIT하는 URL
    path('photo/<int:pk>/edit', views.photo_edit, name="photo_edit"),

    path('detectme', views.detectme, name='detectme'),

    # path('test', views.test, name="test"),

    path('save_photo/', views.save_photo, name='save_photo'),

]