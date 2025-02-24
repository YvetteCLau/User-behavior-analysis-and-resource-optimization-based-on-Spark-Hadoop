from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, hour, avg, unix_timestamp, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# **1. 创建 SparkSession**
spark = SparkSession.builder \
    .appName("User Behavior Clustering & Time Series Analysis") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# **2. 读取数据**
data_path = "file:///home/hadoop/Downloads/图书馆数据/"

entry_logs = spark.read.csv(data_path + "entry_logs.csv", header=True, inferSchema=True, encoding="utf-8")
borrow_logs = spark.read.csv(data_path + "借阅数据_guid_utf8.csv", header=True, inferSchema=True, encoding="utf-8")
seat_logs = spark.read.csv(data_path + "seat_logs.csv", header=True, inferSchema=True, encoding="utf-8")

# **3. 构建用户行为特征**
# 计算入馆次数
user_visit_count = entry_logs.groupBy("ID").agg(count("*").alias("VisitCount"))

# 计算借阅类别分布（确保 `READERID` 改名为 `ID` 以便正确合并）
user_borrow_count = borrow_logs.groupBy("READERID").agg(count("*").alias("BorrowCount")).withColumnRenamed("READERID", "ID")

# 计算座位使用时长
seat_logs = seat_logs.withColumn("UsageTime", 
    (unix_timestamp(col("LeaveSeatTime")) - unix_timestamp(col("SelectSeatTime"))) / 3600)

user_seat_usage = seat_logs.groupBy("ID").agg(avg("UsageTime").alias("AvgUsageTime"))

# **4. 合并数据**
user_features = user_visit_count \
    .join(user_borrow_count, "ID", "left") \
    .join(user_seat_usage, "ID", "left") \
    .fillna({'BorrowCount': 0, 'AvgUsageTime': 0})  # 填充空值

# **5. 进行标准化**
assembler = VectorAssembler(inputCols=["VisitCount", "BorrowCount", "AvgUsageTime"], outputCol="features")
user_features = assembler.transform(user_features)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
user_features = scaler.fit(user_features).transform(user_features)

# **6. KMeans 聚类**
kmeans = KMeans(featuresCol="scaledFeatures", k=3, seed=42)
user_clusters = kmeans.fit(user_features).transform(user_features)

# **7. DBSCAN（使用 BisectingKMeans 近似）**
from pyspark.ml.clustering import BisectingKMeans
dbscan = BisectingKMeans(featuresCol="scaledFeatures", k=3)
user_clusters_dbscan = dbscan.fit(user_features).transform(user_features)

# **8. 时间序列分析（ARIMA）**
# 计算每日入馆次数
daily_visits = entry_logs.withColumn("Date", col("VisitTime").cast("date")) \
                         .groupBy("Date") \
                         .agg(count("*").alias("VisitCount"))

daily_visits_pd = daily_visits.toPandas()
daily_visits_pd.columns = ["ds", "y"]

# 只预测 7 天，减少计算量
arima_model = ARIMA(daily_visits_pd["y"], order=(3,1,0))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=7)

# **9. 学习行为模式分类**
entry_logs = entry_logs.withColumn("Hour", hour(col("VisitTime")))

entry_logs = entry_logs.withColumn("TimeCategory", 
    when((col("Hour") >= 6) & (col("Hour") < 12), "早鸟型")
    .when((col("Hour") >= 18) & (col("Hour") < 24), "夜猫型")
    .otherwise("高频学习型"))

user_behavior_type = entry_logs.groupBy("ID", "TimeCategory").agg(count("*").alias("VisitCount"))

# **10. 用户最终分类**
user_final = user_clusters.select("ID", "prediction").withColumnRenamed("prediction", "KMeansCluster") \
    .join(user_clusters_dbscan.select("ID", "prediction").withColumnRenamed("prediction", "DBSCANCluster"), "ID", "left") \
    .join(user_behavior_type, "ID", "left") \
    .join(user_borrow_count, "ID", "left")  # 确保 `BorrowCount` 存在

# **11. 检查 `user_final` 列名**
print("===== User Final Columns =====")
print(user_final.columns)

# **12. 显示分类数据前几行**
print("===== 用户分类数据（前 5 行）=====")
user_final.show(5)

# **13. 可视化：KMeans 聚类散点图**
user_final_pd = user_final.toPandas()

# 确保 BorrowCount 存在
if "BorrowCount" in user_final_pd.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(user_final_pd["VisitCount"], user_final_pd["BorrowCount"], c=user_final_pd["KMeansCluster"], cmap="viridis", alpha=0.5)
    plt.colorbar(label="Cluster")
    plt.xlabel("Visit Count")
    plt.ylabel("Borrow Count")
    plt.title("User Clustering (KMeans)")
    plt.show()
else:
    print("Error: 'BorrowCount' column is missing in user_final_pd!")

# **14. 可视化：学习行为趋势**
plt.figure(figsize=(8, 5))
plt.scatter(daily_visits_pd["ds"], daily_visits_pd["y"], c="blue", alpha=0.5, label="Historical Data")
plt.scatter(pd.date_range(start=daily_visits_pd["ds"].iloc[-1], periods=7, freq="D"), arima_forecast, c="red", label="Prediction")
plt.legend()
plt.title("User Learning Pattern (ARIMA)")
plt.xlabel("Date")
plt.ylabel("Visit Count")
plt.show()

# **15. 保存结果**
output_path = data_path + "user_clustering/"
user_final.write.csv(output_path + "user_clusters.csv", header=True, mode="overwrite", encoding="utf-8")

print(f"用户行为聚类数据已保存至：{output_path}")

# **16. 结束 SparkSession**
spark.stop()
