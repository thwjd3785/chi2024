from django.urls import path
from . import views

# 나는 static으로 살고 싶은
from django.conf.urls.static import static
# settings에서 설정한 것들
from django.conf import settings
from django.contrib.auth.views import LogoutView

app_name = 'camera_prototype_v2'  # Add this line

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('register/', views.user_register, name='register'),
    path('detectme/', views.detectme, name='detectme'),
    path('get_sensor_data/', views.get_sensor_data, name='get_sensor_data'),
    path('update_sensor_data/', views.update_sensor_data, name='update_sensor_data'),
    path('delete_sensor_data/', views.delete_sensor_data, name='delete_sensor_data'),
    path('save_blocking_area/', views.save_blocking_area, name='save_blocking_area'),
    path('delete_block_data/', views.delete_block_data, name='delete_block_data'),

    path('get_block_data/', views.get_block_data, name='get_block_data'),

    path('stop_camera/', views.stop_camera, name='stop_camera'),

]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
