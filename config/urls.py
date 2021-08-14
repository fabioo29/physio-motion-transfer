from django.contrib import admin
from django.urls import path

from django.conf import settings
from django.conf.urls.static import static

from django.contrib.staticfiles.urls import staticfiles_urlpatterns 

from main import views as main

urlpatterns = [
    path('', main.index, name='index'),
    path('admin/', admin.site.urls),
    path('upload/', main.upload, name='upload'),
    path('clear/', main.clear_main, name='clear_main'),
    path('loading/', main.loadingPage, name='loadingPage'),
    path('getVideo/', main.getVideo, name='getVideo'),
    path('playVideo/', main.playVideo, name='playVideo'),
    path('livefeed/', main.livefe, name='livefe'),
    path('videofeed/', main.videofe, name='videofe'),
] 

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += staticfiles_urlpatterns() 
