from django.urls import include, path

urlpatterns = [
    path('api/', include('myproject.myapp.urls')),  # Adjust 'myapp' based on your app's name
]
