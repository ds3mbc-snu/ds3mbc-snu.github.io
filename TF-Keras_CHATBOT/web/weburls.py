from django.contrib import admin
from django.urls import path, include
from . import webviews

urlpatterns = [
	path('', webviews.index),
]
