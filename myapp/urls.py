from django.urls import path
from . import views 


#URLConf
urlpatterns = [
     path('TextToSpeech/', views.TextToSpeech, name='TextToSpeech'),
     path('TraductionArabic/', views.TraductionArabic, name='TraductionArabic'),
     path('TraductionFrancais/', views.TraductionFrancais, name='TraductionFrancais'),
     path('classification/', views.classification, name='classification'),
     path('result/', views.result, name='result'),
     path('formula_view/', views.formula_view, name='formula_view'),
     path('generate_image/', views.generate_image, name='generate_image'),

     
    path('register/', views.register, name='register'),
    path('', views.login, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.parent_profile_view, name='parent_profile_view'),
    path('commentcreate/', views.create_comment, name='commentcreate'),
    path('comment/', views.comment_list, name='comment'),
     path('add/', views.add_dataset, name='add_dataset'),


]
  