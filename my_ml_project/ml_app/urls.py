from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_image_view, name='predict_image'),
]
