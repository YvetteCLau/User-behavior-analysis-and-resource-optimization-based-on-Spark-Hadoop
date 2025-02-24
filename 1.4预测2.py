from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, unix_timestamp
from pyspark.ml.feature import VectorAssembler, StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# **1. 创建 SparkSession**
spark = SparkSession.builder.appName("User Behavior Prediction").getOrCreate()

# **2. 读取数据**
data_path = "file:///home/hadoop/Downloads/图书馆数据/"
entry_logs = spark.read.csv(data_path + "entry_logs.csv", header=True, inferSchema=True)
borrow_logs = spark.read.csv(data_path + "借阅数据_guid_utf8.csv", header=True, inferSchema=True)
seat_logs = spark.read.csv(data_path + "seat_logs.csv", header=True, inferSchema=True)

# **3. 特征工程**
# 计算入馆次数
user_visit_count = entry_logs.groupBy("ID").agg(count("*").alias("VisitCount"))

# 计算借阅次数
user_borrow_count = borrow_logs.groupBy("READERID").agg(count("*").alias("BorrowCount")).withColumnRenamed("READERID", "ID")

# 计算座位使用时长
seat_logs = seat_logs.withColumn("UsageTime", (unix_timestamp("LeaveSeatTime") - unix_timestamp("SelectSeatTime")) / 3600)
user_seat_usage = seat_logs.groupBy("ID").agg(avg("UsageTime").alias("AvgUsageTime"))

# **4. 合并数据**
user_features = user_visit_count.join(user_borrow_count, "ID", "left").join(user_seat_usage, "ID", "left").fillna(0)

# **5. 转换为 Pandas 并标准化**
user_features_pd = user_features.toPandas()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(user_features_pd[["VisitCount", "BorrowCount", "AvgUsageTime"]])

# **6. 生成时间序列数据**
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(scaled_data, seq_length)

# **7. 划分训练集 & 测试集**
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# **8. 构建 LSTM 模型**
model = Sequential([
    LSTM(32, activation='relu', input_shape=(seq_length, X.shape[2])),
    Dense(X.shape[2], activation='relu')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# **9. 训练 LSTM**
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

# **10. 预测未来 7 天（使用向量化计算）**
future_steps = 7
last_sequence = X_test[-1].reshape(1, seq_length, X.shape[2])
predictions = np.zeros((future_steps, X.shape[2]))

for i in range(future_steps):
    predictions[i] = model.predict(last_sequence, verbose=0)
    last_sequence = np.roll(last_sequence, shift=-1, axis=1)
    last_sequence[0, -1, :] = predictions[i]

predictions = scaler.inverse_transform(predictions)

# **11. 可视化 Loss**
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', c='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', c='red')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("LSTM Model Training Loss")
plt.show()

# **12. 显示未来 7 天预测**
future_dates = pd.date_range(start=user_features_pd.index[-1], periods=future_steps+1)[1:]
print("\nPredicted User Behavior for the Next 7 Days:")
for i, (date, visit, borrow, seat) in enumerate(zip(future_dates, predictions[:, 0], predictions[:, 1], predictions[:, 2])):
    print(f"{date.date()} - Predicted Visits: {int(visit)}, Borrowing: {int(borrow)}, Seat Usage: {seat:.2f} hours")

# **13. 关闭 Spark**
spark.stop()
