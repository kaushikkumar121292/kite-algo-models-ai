from django.urls import path
from .every_minute_ce_views import TrainModelViewCE, PredictViewCE
from .every_minute_pe_views import TrainModelViewPE, PredictViewPE


urlpatterns = [
    path('train-ce/', TrainModelViewCE.as_view(), name='train-model-ce'),
    path('predict-ce/', PredictViewCE.as_view(), name='predict-ce'),
    path('train-pe/', TrainModelViewPE.as_view(), name='train-model-pe'),
    path('predict-pe/', PredictViewPE.as_view(), name='predict-pe'),
]
