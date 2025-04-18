from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_file, name='upload_file'),
    path('analyze_data/', views.analyze_data, name='analyze_data'),
    path('get_graph_data/', views.get_graph_data, name='get_graph_data'),
] 