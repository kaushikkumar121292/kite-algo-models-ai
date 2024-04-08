from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from .model_every_minute_pe import train_model  # Adjust the import based on your file structure
from .model_every_minute_pe import predict_unseen

class TrainModelViewPE(APIView):
    def get(self, request, *args, **kwargs):
        result = train_model()
        return Response(result)


class PredictViewPE(APIView):
    def post(self, request, *args, **kwargs):
        # Manually extract data from request
        raw_unseen_data = request.data

        # Manually validate the data (This is a simplistic approach for demonstration)
        if not isinstance(raw_unseen_data, list) or not all(isinstance(item, dict) for item in raw_unseen_data):
            return JsonResponse({'error': 'Invalid data format. Expected a list of dictionaries.'}, status=400)

        # Assuming each dict has the correct structure and data types
        # In a real application, you'd perform detailed validation here

        predictions = predict_unseen(raw_unseen_data)
        return JsonResponse(predictions)