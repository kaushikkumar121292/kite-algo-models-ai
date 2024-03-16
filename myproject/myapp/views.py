import os

from rest_framework.views import APIView
from rest_framework.response import Response
from . import model_dnn

class TrainModelView(APIView):
    model = None
    scaler = None
    X_test = None
    y_test = None

    def get(self, request, *args, **kwargs):
        # Call the training function which returns the model, scaler, X_test, and y_test
        TrainModelView.model, TrainModelView.scaler, TrainModelView.X_test, TrainModelView.y_test = model_dnn.start_training()

        # Assuming start_training() now returns: model_dnn, scaler, X_test, y_test
        test_accuracy = TrainModelView.model.evaluate(TrainModelView.X_test, TrainModelView.y_test)[1]  # Assuming this is how you get test accuracy

        # Return the test accuracy through the endpoint
        return Response({
            "message": "Model training completed.",
            "test_accuracy": f"{test_accuracy * 100:.2f}%"  # Format as a percentage
        })


class SaveModelView(APIView):
    def get(self, request, *args, **kwargs):
        if TrainModelView.model is not None:
            save_dir = "myproject/myapp/model_dnn_folder"
            save_filename = "model.h5"
            ave_path = os.path.join(save_dir, save_filename)

            # Create the directory if it does not exist
            os.makedirs(save_dir, exist_ok=True)

            # Save the model
            model_dnn.save_model(TrainModelView.model, save_path)
            return Response({"message": "Model saved successfully"})
        else:
            return Response({"message": "No model available to save."}, status=400)

class PredictModelView(APIView):
    def get(self, request, *args, **kwargs):
        # Call predict_unseen_data and capture its return value
        predictions = model_dnn.predict_unseen_data()

        # Convert predictions to a list if they're in a numpy array or similar format
        predictions_list = predictions.tolist() if hasattr(predictions, "tolist") else predictions

        # Return the predictions in the response
        return Response({
            "message": "Prediction done",
            "predictions": predictions_list
        })
