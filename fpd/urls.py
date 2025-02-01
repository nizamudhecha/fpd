from django.contrib import admin
from django.urls import path, include
from . import views  # Import views from the fpd app

urlpatterns = [
    path('', views.Index, name="insta"),
    # path('index', views.Index, name="index"),
    # path("detect/", views.Detect, name="detect"),
   # path('tweet/', views.tweet, name='tweet'),
    #path("twitter/", views.twitter, name="twitter"),
    path('insta/', views.insta, name='insta'),
    path("instagram/", views.instagram, name="instagram"),
    path('about/', views.about, name='about'),  # About Page
    path('contact/', views.contact, name='contact'),  # Contact Page
]
