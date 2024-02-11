import subprocess
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tabulate import tabulate  # Import tabulate library
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping callback


def start_training():
    global your_json_data, all_unseen_features, X_test, y_test, scaler, model_dnn, history
    # Define the curl command to fetch JSON data
    curl_command = [
        'curl',
        '-X',
        'GET',
        '--header',
        'Accept: text/plain',
        'http://kite-algo-zerodha.ap-south-1.elasticbeanstalk.com/trade/download-trade-detail/2024-01-18/ALL'
    ]
    # Execute the curl command to get JSON data
    try:
        json_response = subprocess.check_output(curl_command)
        your_json_data = json.loads(json_response.decode('utf-8'))
    except subprocess.CalledProcessError:
        print("Error executing curl command.")
        your_json_data = []
    # Define a list to store extracted features and targets
    all_features = []  # Modified to store flattened features
    all_targets = []  # Added to store target values
    # Define a list to store extracted features
    all_unseen_features = []
    # Iterate through each data point in your_json_data
    for data_point in your_json_data:
        # Extract features
        ceLegEntry = data_point.get("ceLegEntry", None)
        ceLegTarget = data_point.get("ceLegTarget", None)
        ceLegSl = data_point.get("ceLegSl", None)
        peLegEntry = data_point.get("peLegEntry", None)
        peLegTarget = data_point.get("peLegTarget", None)
        peLegSl = data_point.get("peLegSl", None)

        # Extract and convert dateTime to a numerical representation (timestamp)
        dateTime_str = data_point.get("dateTime", None)
        if dateTime_str:
            dateTime = datetime.strptime(dateTime_str, "%Y-%m-%dT%H:%M:%S.%f").timestamp()
        else:
            dateTime = None

        # Extract oiDifference, oiDivision, volumeDifference, volumeDivision
        oiDifference = data_point.get("oiDifference", None)
        oiDivision = data_point.get("oiDivision", None)
        volumeDifference = data_point.get("volumeDifference", None)
        volumeDivision = data_point.get("volumeDivision", None)

        # Extract day and one-hot encode it
        day = data_point.get("day", None)
        day_encoded = [0, 0, 0, 0, 0, 0, 0]  # Initialize with zeros for all days
        if day:
            day = day.upper()  # Convert to uppercase for consistency
            days_of_week = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
            if day in days_of_week:
                day_index = days_of_week.index(day)
                day_encoded[day_index] = 1

        # Extract option data (CE and PE) dynamically
        option_data_ce = data_point.get("optionDataResponseCe", {}).get("data", {})
        option_data_pe = data_point.get("optionDataResponsePe", {}).get("data", {})

        # Extract attributes for CE option
        ce_option_key = next(iter(option_data_ce), None)  # Get the first key, which is variable
        if ce_option_key:
            ce_data = option_data_ce[ce_option_key]
            last_price_ce = ce_data.get("last_price", None)
            last_quantity_ce = ce_data.get("last_quantity", None)
            average_price_ce = ce_data.get("average_price", None)
            oi_ce = ce_data.get("oi", None)
            oi_day_high_ce = ce_data.get("oi_day_high", None)
            oi_day_low_ce = ce_data.get("oi_day_low", None)
            net_change_ce = ce_data.get("net_change", None)
            lower_circuit_limit_ce = ce_data.get("lower_circuit_limit", None)
            upper_circuit_limit_ce = ce_data.get("upper_circuit_limit", None)
            ohlc_ce = ce_data.get("ohlc", {})

            depth_ce_buy = ce_data.get("depth", {}).get("buy", [])

            depth_ce_first_buy = depth_ce_buy[0] if depth_ce_buy else None
            depth_ce_second_buy = depth_ce_buy[1] if len(depth_ce_buy) > 1 else None
            depth_ce_third_buy = depth_ce_buy[2] if len(depth_ce_buy) > 2 else None
            depth_ce_fourth_buy = depth_ce_buy[3] if len(depth_ce_buy) > 3 else None
            depth_ce_fifth_buy = depth_ce_buy[4] if len(depth_ce_buy) > 4 else None

            depth_ce_sell = ce_data.get("depth", {}).get("sell", [])

            depth_ce_first_sell = depth_ce_sell[0] if depth_ce_sell else None
            depth_ce_second_sell = depth_ce_sell[1] if len(depth_ce_sell) > 1 else None
            depth_ce_third_sell = depth_ce_sell[2] if len(depth_ce_sell) > 2 else None
            depth_ce_fourth_sell = depth_ce_sell[3] if len(depth_ce_sell) > 3 else None
            depth_ce_fifth_sell = depth_ce_sell[4] if len(depth_ce_sell) > 4 else None

            buy_quantity_ce = ce_data.get("buy_quantity", None)
            sell_quantity_ce = ce_data.get("sell_quantity", None)

        else:
            # Handle the case where CE data is not available
            last_price_ce, last_quantity_ce, average_price_ce, oi_ce, oi_day_high_ce, oi_day_low_ce, net_change_ce, lower_circuit_limit_ce, upper_circuit_limit_ce, ohlc_ce, depth_ce_buy = [
                                                                                                                                                                                                None] * 12

        # Extract attributes for PE option
        pe_option_key = next(iter(option_data_pe), None)  # Get the first key, which is variable
        if pe_option_key:
            pe_data = option_data_pe[pe_option_key]
            last_price_pe = pe_data.get("last_price", None)
            last_quantity_pe = pe_data.get("last_quantity", None)
            average_price_pe = pe_data.get("average_price", None)
            oi_pe = pe_data.get("oi", None)
            oi_day_high_pe = pe_data.get("oi_day_high", None)
            oi_day_low_pe = pe_data.get("oi_day_low", None)
            net_change_pe = pe_data.get("net_change", None)
            lower_circuit_limit_pe = pe_data.get("lower_circuit_limit", None)
            upper_circuit_limit_pe = pe_data.get("upper_circuit_limit", None)
            ohlc_pe = pe_data.get("ohlc", {})

            depth_pe_buy = pe_data.get("depth", {}).get("buy", [])

            depth_pe_buy = pe_data.get("depth", {}).get("buy", [])

            depth_pe_first_buy = depth_pe_buy[0] if depth_pe_buy else None
            depth_pe_second_buy = depth_pe_buy[1] if len(depth_pe_buy) > 1 else None
            depth_pe_third_buy = depth_pe_buy[2] if len(depth_pe_buy) > 2 else None
            depth_pe_fourth_buy = depth_pe_buy[3] if len(depth_pe_buy) > 3 else None
            depth_pe_fifth_buy = depth_pe_buy[4] if len(depth_pe_buy) > 4 else None

            depth_pe_sell = pe_data.get("depth", {}).get("sell", [])

            depth_pe_first_sell = depth_pe_sell[0] if depth_pe_sell else None
            depth_pe_second_sell = depth_pe_sell[1] if len(depth_pe_sell) > 1 else None
            depth_pe_third_sell = depth_pe_sell[2] if len(depth_pe_sell) > 2 else None
            depth_pe_fourth_sell = depth_pe_sell[3] if len(depth_pe_sell) > 3 else None
            depth_pe_fifth_sell = depth_pe_sell[4] if len(depth_pe_sell) > 4 else None

            buy_quantity_pe = pe_data.get("buy_quantity", None)
            sell_quantity_pe = pe_data.get("sell_quantity", None)
        else:
            # Handle the case where PE data is not available
            last_price_pe, last_quantity_pe, average_price_pe, oi_pe, oi_day_high_pe, oi_day_low_pe, net_change_pe, lower_circuit_limit_pe, upper_circuit_limit_pe, ohlc_pe, depth_pe = [
                                                                                                                                                                                            None] * 12

        # Extract attributes from candleResponseCe and candleResponsePe if available
        candle_response_ce = data_point.get("candleResponseCe", {}).get("data", {}).get("candles", [])
        candle_response_pe = data_point.get("candleResponsePe", {}).get("data", {}).get("candles", [])

        # Append all the extracted features to the 'features' list
        features = [
            ceLegEntry, ceLegTarget, ceLegSl, peLegEntry, peLegTarget, peLegSl, oiDifference, oiDivision,
            volumeDifference, volumeDivision, dateTime,
            last_price_ce, last_quantity_ce, average_price_ce, oi_ce, oi_day_high_ce, oi_day_low_ce,
            net_change_ce, lower_circuit_limit_ce, upper_circuit_limit_ce,
            last_price_pe, last_quantity_pe, average_price_pe, oi_pe, oi_day_high_pe, oi_day_low_pe,
            net_change_pe, lower_circuit_limit_pe, upper_circuit_limit_pe, buy_quantity_ce, sell_quantity_ce,
            buy_quantity_pe, sell_quantity_pe
        ]
        features.extend(day_encoded)

        # Append "ohlc" and "depth" for CE and PE options

        features.extend(ohlc_ce.values())

        depth_variables = [depth_ce_first_buy, depth_ce_second_buy, depth_ce_third_buy, depth_ce_fourth_buy,
                           depth_ce_fifth_buy,
                           depth_ce_first_sell, depth_ce_second_sell, depth_ce_third_sell, depth_ce_fourth_sell,
                           depth_ce_fifth_sell,
                           depth_pe_first_buy, depth_pe_second_buy, depth_pe_third_buy, depth_pe_fourth_buy,
                           depth_pe_fifth_buy,
                           depth_pe_first_sell, depth_pe_second_sell, depth_pe_third_sell, depth_pe_fourth_sell,
                           depth_pe_fifth_sell]

        for depth_variable in depth_variables:
            if depth_variable:
                features.extend(depth_variable.values())
            else:
                # If the depth variable is None, extend with None values
                features.extend([None] * len(depth_variable))

        features.extend(ohlc_pe.values())

        # Append candle_response_ce and candle_response_pe

        if candle_response_ce:
            first_candle_ce = candle_response_ce[0]
            features.extend(first_candle_ce[1:])  # Append all elements from the second element onwards

        if candle_response_pe:
            first_candle_pe = candle_response_pe[0]
            features.extend(first_candle_pe[1:])  # Append all elements from the second element onwards

        all_features.append(features)

        # Append 'success' (target variable) to targets
        success = data_point.get("success", None)
        all_targets.append(success)
    # Convert your feature and target lists to NumPy arrays for compatibility with TensorFlow
    X = np.array(all_features)
    y = np.array(all_targets)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Apply Tomek Links for undersampling
    tomek = TomekLinks()
    X_resampled, y_resampled = tomek.fit_resample(X_train, y_train)
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_resampled)  # Use resampled data for training
    X_test = scaler.transform(X_test)
    model_dnn = tf.keras.Sequential([
        tf.keras.layers.Input(shape=X_train.shape[1]),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Adding dropout for regularization
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification, so use 'sigmoid' activation
    ])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # Compile the model
    model_dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    history = model_dnn.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
                            callbacks=[early_stopping])
    # Evaluate the model on the test data
    loss, test_accuracy = model_dnn.evaluate(X_test, y_test)
    # Get training and validation accuracy from the training history
    training_accuracy = history.history['accuracy'][-1]
    validation_accuracy = history.history['val_accuracy'][-1]
    # Print the accuracies
    print(f'Training Accuracy: ~{training_accuracy * 100:.2f}%')
    print(f'Validation Accuracy: ~{validation_accuracy * 100:.2f}%')
    print(f'Test Accuracy: ~{test_accuracy * 100:.2f}%')

    return model_dnn, scaler, X_test, y_test


start_training()

y_pred = model_dnn.predict(X_test)

# Convert predicted values to binary (0 or 1) using a threshold (e.g., 0.5)
threshold = 0.5
y_pred_binary = (y_pred > threshold).astype(int)

# Create a table to display actual and predicted values
table_data = [["Actual Values", *y_test],
              ["Predicted Values (Binary)", *y_pred_binary]]

# Print the table
table_headers = [""] + [f"Sample {i+1}" for i in range(len(y_test))]
table = tabulate(table_data, headers=table_headers, tablefmt="grid")
print(table)


# Define the function to preprocess the data
def preprocess_data(data_point):
    for data_point in your_json_data:
        # Extract features
        ceLegEntry = data_point.get("ceLegEntry", None)
        ceLegTarget = data_point.get("ceLegTarget", None)
        ceLegSl = data_point.get("ceLegSl", None)
        peLegEntry = data_point.get("peLegEntry", None)
        peLegTarget = data_point.get("peLegTarget", None)
        peLegSl = data_point.get("peLegSl", None)

        # Extract and convert dateTime to a numerical representation (timestamp)
        dateTime_str = data_point.get("dateTime", None)
        if dateTime_str:
            dateTime = datetime.strptime(dateTime_str, "%Y-%m-%dT%H:%M:%S.%f").timestamp()
        else:
            dateTime = None

        # Extract oiDifference, oiDivision, volumeDifference, volumeDivision
        oiDifference = data_point.get("oiDifference", None)
        oiDivision = data_point.get("oiDivision", None)
        volumeDifference = data_point.get("volumeDifference", None)
        volumeDivision = data_point.get("volumeDivision", None)

        # Extract day and one-hot encode it
        day = data_point.get("day", None)
        day_encoded = [0, 0, 0, 0, 0, 0, 0]  # Initialize with zeros for all days
        if day:
            day = day.upper()  # Convert to uppercase for consistency
            days_of_week = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
            if day in days_of_week:
                day_index = days_of_week.index(day)
                day_encoded[day_index] = 1

        # Extract option data (CE and PE) dynamically
        option_data_ce = data_point.get("optionDataResponseCe", {}).get("data", {})
        option_data_pe = data_point.get("optionDataResponsePe", {}).get("data", {})

        # Extract attributes for CE option
        ce_option_key = next(iter(option_data_ce), None)  # Get the first key, which is variable
        if ce_option_key:
            ce_data = option_data_ce[ce_option_key]
            last_price_ce = ce_data.get("last_price", None)
            last_quantity_ce = ce_data.get("last_quantity", None)
            average_price_ce = ce_data.get("average_price", None)
            oi_ce = ce_data.get("oi", None)
            oi_day_high_ce = ce_data.get("oi_day_high", None)
            oi_day_low_ce = ce_data.get("oi_day_low", None)
            net_change_ce = ce_data.get("net_change", None)
            lower_circuit_limit_ce = ce_data.get("lower_circuit_limit", None)
            upper_circuit_limit_ce = ce_data.get("upper_circuit_limit", None)
            ohlc_ce = ce_data.get("ohlc", {})

            depth_ce_buy = ce_data.get("depth", {}).get("buy", [])

            depth_ce_first_buy = depth_ce_buy[0] if depth_ce_buy else None
            depth_ce_second_buy = depth_ce_buy[1] if len(depth_ce_buy) > 1 else None
            depth_ce_third_buy = depth_ce_buy[2] if len(depth_ce_buy) > 2 else None
            depth_ce_fourth_buy = depth_ce_buy[3] if len(depth_ce_buy) > 3 else None
            depth_ce_fifth_buy = depth_ce_buy[4] if len(depth_ce_buy) > 4 else None

            depth_ce_sell = ce_data.get("depth", {}).get("sell", [])

            depth_ce_first_sell = depth_ce_sell[0] if depth_ce_sell else None
            depth_ce_second_sell = depth_ce_sell[1] if len(depth_ce_sell) > 1 else None
            depth_ce_third_sell = depth_ce_sell[2] if len(depth_ce_sell) > 2 else None
            depth_ce_fourth_sell = depth_ce_sell[3] if len(depth_ce_sell) > 3 else None
            depth_ce_fifth_sell = depth_ce_sell[4] if len(depth_ce_sell) > 4 else None

            buy_quantity_ce = ce_data.get("buy_quantity", None)
            sell_quantity_ce = ce_data.get("sell_quantity", None)

        else:
            # Handle the case where CE data is not available
            last_price_ce, last_quantity_ce, average_price_ce, oi_ce, oi_day_high_ce, oi_day_low_ce, net_change_ce, lower_circuit_limit_ce, upper_circuit_limit_ce, ohlc_ce, depth_ce_buy = [
                                                                                                                                                                                                None] * 12

        # Extract attributes for PE option
        pe_option_key = next(iter(option_data_pe), None)  # Get the first key, which is variable
        if pe_option_key:
            pe_data = option_data_pe[pe_option_key]
            last_price_pe = pe_data.get("last_price", None)
            last_quantity_pe = pe_data.get("last_quantity", None)
            average_price_pe = pe_data.get("average_price", None)
            oi_pe = pe_data.get("oi", None)
            oi_day_high_pe = pe_data.get("oi_day_high", None)
            oi_day_low_pe = pe_data.get("oi_day_low", None)
            net_change_pe = pe_data.get("net_change", None)
            lower_circuit_limit_pe = pe_data.get("lower_circuit_limit", None)
            upper_circuit_limit_pe = pe_data.get("upper_circuit_limit", None)
            ohlc_pe = pe_data.get("ohlc", {})

            depth_pe_buy = pe_data.get("depth", {}).get("buy", [])

            depth_pe_buy = pe_data.get("depth", {}).get("buy", [])

            depth_pe_first_buy = depth_pe_buy[0] if depth_pe_buy else None
            depth_pe_second_buy = depth_pe_buy[1] if len(depth_pe_buy) > 1 else None
            depth_pe_third_buy = depth_pe_buy[2] if len(depth_pe_buy) > 2 else None
            depth_pe_fourth_buy = depth_pe_buy[3] if len(depth_pe_buy) > 3 else None
            depth_pe_fifth_buy = depth_pe_buy[4] if len(depth_pe_buy) > 4 else None

            depth_pe_sell = pe_data.get("depth", {}).get("sell", [])

            depth_pe_first_sell = depth_pe_sell[0] if depth_pe_sell else None
            depth_pe_second_sell = depth_pe_sell[1] if len(depth_pe_sell) > 1 else None
            depth_pe_third_sell = depth_pe_sell[2] if len(depth_pe_sell) > 2 else None
            depth_pe_fourth_sell = depth_pe_sell[3] if len(depth_pe_sell) > 3 else None
            depth_pe_fifth_sell = depth_pe_sell[4] if len(depth_pe_sell) > 4 else None

            buy_quantity_pe = pe_data.get("buy_quantity", None)
            sell_quantity_pe = pe_data.get("sell_quantity", None)
        else:
            # Handle the case where PE data is not available
            last_price_pe, last_quantity_pe, average_price_pe, oi_pe, oi_day_high_pe, oi_day_low_pe, net_change_pe, lower_circuit_limit_pe, upper_circuit_limit_pe, ohlc_pe, depth_pe = [
                                                                                                                                                                                            None] * 12

        # Extract attributes from candleResponseCe and candleResponsePe if available
        candle_response_ce = data_point.get("candleResponseCe", {}).get("data", {}).get("candles", [])
        candle_response_pe = data_point.get("candleResponsePe", {}).get("data", {}).get("candles", [])

        # Append all the extracted features to the 'features' list
        features = [
            ceLegEntry, ceLegTarget, ceLegSl, peLegEntry, peLegTarget, peLegSl, oiDifference, oiDivision,
            volumeDifference, volumeDivision, dateTime,
            last_price_ce, last_quantity_ce, average_price_ce, oi_ce, oi_day_high_ce, oi_day_low_ce,
            net_change_ce, lower_circuit_limit_ce, upper_circuit_limit_ce,
            last_price_pe, last_quantity_pe, average_price_pe, oi_pe, oi_day_high_pe, oi_day_low_pe,
            net_change_pe, lower_circuit_limit_pe, upper_circuit_limit_pe, buy_quantity_ce, sell_quantity_ce,
            buy_quantity_pe, sell_quantity_pe
        ]
        features.extend(day_encoded)

        # Append "ohlc" and "depth" for CE and PE options

        features.extend(ohlc_ce.values())

        depth_variables = [depth_ce_first_buy, depth_ce_second_buy, depth_ce_third_buy, depth_ce_fourth_buy,
                           depth_ce_fifth_buy,
                           depth_ce_first_sell, depth_ce_second_sell, depth_ce_third_sell, depth_ce_fourth_sell,
                           depth_ce_fifth_sell,
                           depth_pe_first_buy, depth_pe_second_buy, depth_pe_third_buy, depth_pe_fourth_buy,
                           depth_pe_fifth_buy,
                           depth_pe_first_sell, depth_pe_second_sell, depth_pe_third_sell, depth_pe_fourth_sell,
                           depth_pe_fifth_sell]

        for depth_variable in depth_variables:
            if depth_variable:
                features.extend(depth_variable.values())
            else:
                # If the depth variable is None, extend with None values
                features.extend([None] * len(depth_variable))

        features.extend(ohlc_pe.values())

        # Append candle_response_ce and candle_response_pe

        if candle_response_ce:
            first_candle_ce = candle_response_ce[0]
            features.extend(first_candle_ce[1:])  # Append all elements from the second element onwards

        if candle_response_pe:
            first_candle_pe = candle_response_pe[0]
            features.extend(first_candle_pe[1:])  # Append all elements from the second element onwards

        return features


def predict_unseen_data():
    global threshold
    all_unseen_features = []  # Initialize the list to store preprocessed features
    threshold = 0.5  # Define the threshold for binary classification

    # Define the curl command to fetch JSON data for unseen data
    curl_command_unseen = [
        'curl',
        '-X',
        'GET',
        '--header',
        'Accept: text/plain',
        'http://kite-algo-zerodha.ap-south-1.elasticbeanstalk.com/trade/download-active-trade-detail-unseen'
    ]

    try:
        # Execute the curl command to get JSON data for unseen data
        json_response_unseen = subprocess.check_output(curl_command_unseen)
        unseen_data = json.loads(json_response_unseen.decode('utf-8'))

        # Iterate through each data point in unseen_data
        for data_point_unseen in unseen_data:
            # Preprocess the data
            features_unseen = preprocess_data(data_point_unseen)
            all_unseen_features.append(features_unseen)

        # Convert the feature list to a NumPy array
        X_unseen = np.array(all_unseen_features)

        # Normalize features using the same StandardScaler used for training
        X_unseen = scaler.transform(X_unseen)

        # Make predictions on the unseen data
        y_unseen_pred = model_dnn.predict(X_unseen)

        # Convert predicted values to binary (0 or 1) using a threshold
        y_unseen_pred_binary = (y_unseen_pred > threshold).astype(int)

        # Print the predicted binary values
        print("Predicted Values (Binary) for Unseen Data:")
        print(y_unseen_pred_binary)

    except subprocess.CalledProcessError:
        print("Error executing curl command for unseen data.")
        y_unseen_pred_binary = []
    except Exception as e:
        print(f"An error occurred during the prediction process: {e}")
        y_unseen_pred_binary = []

    return y_unseen_pred_binary


predict_unseen_data()


def save_model(model, save_path="model_dnn"):
    model.save(save_path)
    print(f"Model saved to {save_path}")
