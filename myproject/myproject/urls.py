from django.urls import include, path

urlpatterns = [
    # ... other url patterns
    path('api/', include('myproject.myapp.urls')),
]
