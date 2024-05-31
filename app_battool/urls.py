from django.urls import path
from . import views

urlpatterns = [
    path('',views.home,name=''),
    path('result',views.process_audio,name='result'),
    path('inputview',views.input_view,name='inputview'),
    path('add_species',views.add_species,name='add_species'),
]