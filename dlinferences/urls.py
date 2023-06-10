from django.urls import path
from . import views

urlpatterns = [
    path("facial-expression/predict/", views.facial_expression_analysis, name="facial_expression_analysis")
]