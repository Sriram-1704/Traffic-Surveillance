from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path('UserLogin', views.UserLogin, name="UserLogin"),
	       path('UserLoginAction', views.UserLoginAction, name="UserLoginAction"),	   
	       path('Signup', views.Signup, name="Signup"),
	       path('SignupAction', views.SignupAction, name="SignupAction"),
	       path('TrainYolo', views.TrainYolo, name="TrainYolo"),
	       path('DetectTraffic', views.DetectTraffic, name="DetectTraffic"),
	       path('DetectTrafficAction', views.DetectTrafficAction, name="DetectTrafficAction"),	
	       path('Aboutus', views.Aboutus, name="Aboutus"),
	       
]