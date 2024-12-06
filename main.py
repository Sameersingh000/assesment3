import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare the dataset
data = pd.read_csv('merged_data.csv')

# Define input features (X) and target output (y)
X = data[['Signed Volume', 'best_bid', 'best_ask', 'mid_price']].fillna(0)  # Replace NaN with 0
y = data['Signed Volume'].fillna(0)  # Use Signed Volume as target

# Normalize the features for consistent scale
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Define a function to create and train a neural network
def train_dnn(hidden_layers, units_per_layer, epochs=50, batch_size=32):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units_per_layer, activation='relu', input_shape=(X_train.shape[1],)))
    
    # Add hidden layers
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(units_per_layer, activation='relu'))
    
    # Output layer
    model.add(tf.keras.layers.Dense(1))  # Single output for trading action
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size, verbose=0)
    return history

# Train networks with different structures
network_structures = [
    {"hidden_layers": 1, "units_per_layer": 32},
    {"hidden_layers": 2, "units_per_layer": 64},
    {"hidden_layers": 3, "units_per_layer": 128}
]

loss_results = {}

for structure in network_structures:
    print(f"Training network: {structure}")
    history = train_dnn(structure["hidden_layers"], structure["units_per_layer"])
    loss_results[f"{structure['hidden_layers']} Layers, {structure['units_per_layer']} Units"] = history.history['loss']

# Visualize the training loss for different structures
plt.figure(figsize=(10, 6))
for label, loss in loss_results.items():
    plt.plot(loss, label=label)

plt.xlabel("Epochs")
plt.ylabel("Training Loss (MSE)")
plt.title("Training Loss for Different Network Structures")
plt.legend()
plt.grid(True)
plt.show()
