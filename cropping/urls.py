from django.urls import path
from . import views

# 나는 static으로 살고 싶은
from django.conf.urls.static import static
# settings에서 설정한 것들
from django.conf import settings

urlpatterns = [
    path('', views.index, name='index'),
    #path('save_image/', save_image, name='save_image'),
    path('detectme/', views.detectme, name='detectme'),
    path('detectme/object_detection/', views.object_detection, name='object_detection'),

]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
