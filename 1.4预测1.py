import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# **1. Load Borrowing Data**
data_path = "file:///home/hadoop/Downloads/Library_Data/"
df = pd.read_csv(data_path + "borrowing_data_utf8.csv")

# **2. Standardize Column Names**
df.columns = [col.lower() for col in df.columns]

# **3. Ensure 'lenddate' exists**
if "lenddate" not in df.columns or "readerid" not in df.columns:
    raise ValueError("CSV file is missing 'LendDate' or 'ReaderID' column!")

# **4. Process Date Format**
df["lenddate"] = pd.to_datetime(df["lenddate"], errors="coerce")
df = df.dropna(subset=["lenddate"])
df["date"] = df["lenddate"].dt.date

# **5. Calculate Daily Borrowing Count**
daily_borrows = df.groupby("date").size().reset_index(name="borrowcount")
daily_borrows.set_index("date", inplace=True)

# **6. Normalize Data**
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(daily_borrows)

# **7. Create Time Window Sequences**
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 14
X, y = create_sequences(scaled_data, seq_length)

# **8. Split Training & Testing Sets**
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# **9. Build LSTM Model**
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, activation='relu'),
    Dense(1, activation='relu')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# **10. Train LSTM**
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test),
                    callbacks=[early_stopping], verbose=1)

# **11. Predict the Next 7 Days**
future_steps = 7
predictions = []
last_sequence = X_test[-1]

for _ in range(future_steps):
    next_prediction = model.predict(last_sequence.reshape(1, seq_length, 1))
    predictions.append(next_prediction[0])
    last_sequence = np.vstack([last_sequence[1:], next_prediction])

predictions = scaler.inverse_transform(predictions)

# **12. Generate Future Dates**
future_dates = pd.date_range(start=daily_borrows.index[-1], periods=future_steps+1)[1:]

# **13. Plot Borrowing Predictions**
plt.figure(figsize=(12, 5))
plt.plot(daily_borrows.index[-len(y_test):], scaler.inverse_transform(y_test), label="Actual Borrow Count", c="blue")
plt.plot(future_dates, predictions, label="Predicted Borrow Count", linestyle="dashed", c="red")
plt.legend()
plt.title("Library Borrowing Prediction (LSTM)")
plt.xlabel("Date")
plt.ylabel("Borrow Count")
plt.xticks(rotation=45)
plt.show()

# **14. Display Predicted Data**
print("\nPredicted Borrowing for the Next 7 Days:")
for i, (date, borrow) in enumerate(zip(future_dates, predictions)):
    print(f"{date.date()} - Predicted Borrowing: {int(borrow)} books")
    if borrow > np.percentile(daily_borrows["borrowcount"], 90):
        print("Recommendation: Increase stock for popular books.")

# **15. Visualize Training Loss**
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', c='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', c='red')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("LSTM Model Training Loss")
plt.show()

# **16. Calculate Borrowing Per User**
user_borrows = df.groupby(["date", "readerid"]).size().reset_index(name="user_borrowcount")
user_borrows = user_borrows.pivot(index="date", columns="readerid", values="user_borrowcount").fillna(0)

# **17. Select Top 10 Users for Prediction**
selected_users = user_borrows.columns[:10]
plt.figure(figsize=(12, 6))

for user in selected_users:
    user_data = user_borrows[user].values.reshape(-1, 1)
    user_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_user_data = user_scaler.fit_transform(user_data)

    X_user, y_user = create_sequences(scaled_user_data, seq_length)
    X_user_train, y_user_train = X_user[:train_size], y_user[:train_size]
    X_user_test, y_user_test = X_user[train_size:], y_user[train_size:]

    # Train LSTM for User Borrowing Prediction
    user_model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50, activation='relu'),
        Dense(1, activation='relu')
    ])
    user_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    user_model.fit(X_user_train, y_user_train, epochs=20, batch_size=16, verbose=0)

    # Predict Next 7 Days for the User
    user_predictions = []
    last_sequence = X_user_test[-1]

    for _ in range(future_steps):
        next_prediction = user_model.predict(last_sequence.reshape(1, seq_length, 1))
        user_predictions.append(next_prediction[0])
        last_sequence = np.vstack([last_sequence[1:], next_prediction])

    user_predictions = user_scaler.inverse_transform(user_predictions)

    # **18. Plot User Borrowing Predictions**
    plt.plot(future_dates, user_predictions, label=f"User {user}")

    # **19. Display User Predictions**
    print(f"\nPredicted Borrowing for User {user} in the Next 7 Days:")
    for i, (date, borrow) in enumerate(zip(future_dates, user_predictions)):
        print(f"{date.date()} - Predicted Borrowing: {int(borrow)} books")

plt.legend()
plt.title("Top 10 Users' Borrowing Trend Prediction")
plt.xlabel("Date")
plt.ylabel("Borrow Count")
plt.xticks(rotation=45)
plt.show()
