import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import TomekLinks
from tabulate import tabulate
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Fetch the data
url = 'http://localhost:5000/trade/download-every-minute-training-data-v1-ce'
response = requests.get(url, headers={'Accept': 'text/plain'})
data = response.json()  # Adjust this if the data format is different

# Example of how you might process the fetched data
# Assuming the data is a list of dictionaries with features and a 'success' key for the binary target
features_list = []
target_list = []

for record in data:
    # Assuming 'optionDataResponse' is not included in your feature set
    features = [record['indiaVix'], record['theta'],record['oi_change'], record['max_oi'], record['ltp_change'], record['delta'], record['gamma'], record['vega'], record['iv'], record['last_price'], record['oi'], record['volume'], record['atm_iv'], record['future_price'], record['underlying_price']]
    features_list.append(features)
    target_list.append(1 if record['success'] else 0)

# Convert lists to numpy arrays
X = np.array(features_list)
y = np.array(target_list)

# Continue with preprocessing...
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Applying Tomek Links for balancing the dataset
tl = TomekLinks()
X_resampled, y_resampled = tl.fit_resample(X_train, y_train)

# Define the DNN model
model = Sequential([
    Dense(1024, activation='relu', input_shape=(X_resampled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train the model with early stopping
model.fit(X_resampled, y_resampled, epochs=100, validation_split=0.2, callbacks=[early_stopping], batch_size=32)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Assuming predictions and y_test are defined as before
predictions_proba = model.predict(X_test)
predictions = (predictions_proba > 0.5).astype(int)

# Create a list of lists where each sub-list is a pair of actual and predicted values
table_data = list(zip(y_test, predictions.flatten()))

# Display in table format using tabulate
print(tabulate(table_data, headers=['Actual', 'Predicted'], tablefmt='grid', showindex="always"))

# Calculate Profit
profit_for_true_positive = 15  # Actual 1, Predicted 1
loss_for_false_positive = -5   # Actual 0, Predicted 1
total_profit = 0

for actual, predicted in zip(y_test, predictions.flatten()):
    if actual == 1 and predicted == 1:
        total_profit += profit_for_true_positive
    elif actual == 0 and predicted == 1:
        total_profit += loss_for_false_positive

print(f'Total Profit: {total_profit} Rupees')

