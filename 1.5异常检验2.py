from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from sklearn.ensemble import IsolationForest

# **1. 创建 SparkSession（增加内存）**
spark = SparkSession.builder \
    .appName("Library Access Analysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# **2. 读取门禁数据**
data_path = "file:///home/hadoop/Downloads/图书馆数据/"
entry_logs = spark.read.csv(data_path + "entry_logs.csv", header=True, inferSchema=True)

# **3. 处理入馆记录**
entry_logs = entry_logs.withColumn("Hour", hour(col("VisitTime"))) \
                       .withColumn("Weekday", dayofweek(col("VisitTime")))

# **4. 异常行为检测（Isolation Forest）**
# ✅ **限制数据量，避免 `.toPandas()` 过载**
access_data = entry_logs.select("ID", "Hour").limit(10000).toPandas()

iso_forest = IsolationForest(contamination=0.05, random_state=42)
access_data["AnomalyScore"] = iso_forest.fit_predict(access_data[["Hour"]])

anomaly_df = spark.createDataFrame(access_data)

# **5. 用户行为聚类（KMeans）**
assembler = VectorAssembler(inputCols=["Hour", "Weekday"], outputCol="features")
feature_data = assembler.transform(entry_logs.limit(10000))  # ✅ **限制数据量**
kmeans = KMeans(featuresCol="features", k=3, seed=42)
user_clusters = kmeans.fit(feature_data).transform(feature_data)

# **6. 展示结果**
print("\n📌 Top 10 - 异常行为用户：")
anomaly_df.show(10)

print("\n📌 Top 10 - 用户行为聚类：")
user_clusters.select("ID", "prediction").show(10)

# **7. 存储分析结果**
output_path = "file:///home/hadoop/Downloads/分析结果/"

anomaly_df.write.csv(output_path + "anomalies.csv", header=True, mode="overwrite")
user_clusters.select("ID", "prediction").write.csv(output_path + "user_clusters.csv", header=True, mode="overwrite")

# **8. 关闭 Spark**
spark.stop()
