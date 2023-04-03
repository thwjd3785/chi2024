from django.urls import path
from .views import index, save_image

# 나는 static으로 살고 싶은
from django.conf.urls.static import static
# settings에서 설정한 것들
from django.conf import settings

urlpatterns = [
    path('', index, name='index'),
    #path('save_image/', save_image, name='save_image'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
