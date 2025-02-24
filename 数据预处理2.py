from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, when, hour, lit, expr, max
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 创建 SparkSession
spark = SparkSession.builder \
    .appName("Library User Behavior Analysis") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.default.parallelism", "200") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .getOrCreate()

# 数据文件路径
data_path = "file:///home/hadoop/Downloads/图书馆数据/"

# 读取数据并转换为 Parquet 格式
reader_data = spark.read.csv(data_path + "读者数据_guid_utf8.csv", header=True, inferSchema=True, encoding="utf-8")
entry_logs = spark.read.csv(data_path + "entry_logs.csv", header=True, inferSchema=True, encoding="utf-8")
borrow_logs = spark.read.csv(data_path + "借阅数据_guid_utf8.csv", header=True, inferSchema=True, encoding="utf-8")
seat_logs = spark.read.csv(data_path + "seat_logs.csv", header=True, inferSchema=True, encoding="utf-8")
book_data = spark.read.csv(data_path + "图书数据_utf8.csv", header=True, inferSchema=True, encoding="utf-8")

# 转换为 Parquet 格式
reader_data.write.parquet(data_path + "读者数据_guid_utf8.parquet", mode="overwrite")
entry_logs.write.parquet(data_path + "entry_logs.parquet", mode="overwrite")
borrow_logs.write.parquet(data_path + "借阅数据_guid_utf8.parquet", mode="overwrite")
seat_logs.write.parquet(data_path + "seat_logs.parquet", mode="overwrite")
book_data.write.parquet(data_path + "图书数据_utf8.parquet", mode="overwrite")

# 重新读取 Parquet 数据
reader_data = spark.read.parquet(data_path + "读者数据_guid_utf8.parquet")
entry_logs = spark.read.parquet(data_path + "entry_logs.parquet")
borrow_logs = spark.read.parquet(data_path + "借阅数据_guid_utf8.parquet")
seat_logs = spark.read.parquet(data_path + "seat_logs.parquet")
book_data = spark.read.parquet(data_path + "图书数据_utf8.parquet")

# 重命名列，避免歧义
reader_data = reader_data.withColumnRenamed("ID", "reader_id")
entry_logs = entry_logs.withColumnRenamed("ID", "entry_id")
borrow_logs = borrow_logs.withColumnRenamed("READERID", "borrow_reader_id")
seat_logs = seat_logs.withColumnRenamed("ID", "seat_id")

# **两两融合：读者信息结合其他表**
# 1. 读者信息与门禁记录融合
reader_entry_data = reader_data.join(
    entry_logs,
    reader_data["reader_id"] == entry_logs["entry_id"],
    how="left"
)

# 2. 读者信息与借阅记录融合
reader_borrow_data = reader_data.join(
    borrow_logs,
    reader_data["reader_id"] == borrow_logs["borrow_reader_id"],
    how="left"
)

# 3. 读者信息与座位记录融合
reader_seat_data = reader_data.join(
    seat_logs,
    reader_data["reader_id"] == seat_logs["seat_id"],
    how="left"
)

# **特征工程**
# 特征 1：统计借阅时间段（早上、下午、晚上）的次数
reader_borrow_data = reader_borrow_data.withColumn(
    "borrow_time_period",
    when((hour(col("BORROW_TIME")) >= 6) & (hour(col("BORROW_TIME")) < 12), lit("Morning"))
    .when((hour(col("BORROW_TIME")) >= 12) & (hour(col("BORROW_TIME")) < 18), lit("Afternoon"))
    .when((hour(col("BORROW_TIME")) >= 18) & (hour(col("BORROW_TIME")) < 24), lit("Evening"))
    .otherwise(lit("Night"))
)

borrow_time_distribution = reader_borrow_data.groupBy("reader_id", "borrow_time_period").count().groupBy("reader_id").pivot("borrow_time_period").sum("count").na.fill(0)

# 特征 2：统计座位使用区域（假设有列 REGION 表示座位区域）
seat_region_distribution = reader_seat_data.groupBy("reader_id", "REGION").count().groupBy("reader_id").pivot("REGION").sum("count").na.fill(0)

# 特征 3：加入用户行为画像
user_features = reader_entry_data.groupBy("reader_id").agg(
    count("entry_id").alias("entry_count"),  # 入馆次数
).join(
    reader_borrow_data.groupBy("reader_id").agg(
        count("borrow_reader_id").alias("borrow_count"),  # 借阅次数
    ),
    on="reader_id",
    how="left"
).join(
    reader_seat_data.groupBy("reader_id").agg(
        count("seat_id").alias("seat_usage_count"),  # 座位使用次数
        sum("DURATION").alias("total_seat_duration"),  # 总座位使用时长
        avg("DURATION").alias("avg_seat_duration"),  # 平均座位使用时长
    ),
    on="reader_id",
    how="left"
).join(
    borrow_time_distribution,
    on="reader_id",
    how="left"
).join(
    seat_region_distribution,
    on="reader_id",
    how="left"
).na.fill(0)

# 展示前 50 名用户的行为画像
print("\n前 50 名用户的行为画像：")
user_features.show(50)

# **聚类分析**
feature_columns = ["entry_count", "borrow_count", "seat_usage_count", "total_seat_duration", "avg_seat_duration"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
user_features_vector = assembler.transform(user_features)

kmeans = KMeans(k=3, seed=1, featuresCol="features", predictionCol="cluster")
model = kmeans.fit(user_features_vector)
clustered_data = model.transform(user_features_vector)

# **可视化：柱状图、饼图、散点图**
sns.set_palette("pastel")

# 1. 柱状图（Top 10 入馆次数）
top_10_entry = user_features.orderBy(col("entry_count").desc()).limit(10).toPandas()
plt.figure(figsize=(10, 6))
sns.barplot(x="reader_id", y="entry_count", data=top_10_entry, palette="Blues_d")
plt.title("Top 10 Readers by Entry Count")
plt.xlabel("Reader ID")
plt.ylabel("Entry Count")
plt.xticks(rotation=45)
plt.show()

# 2. 饼图（借阅时间段分布）
borrow_time_sum = borrow_time_distribution.select(sum("Morning").alias("Morning"),
                                                  sum("Afternoon").alias("Afternoon"),
                                                  sum("Evening").alias("Evening"),
                                                  sum("Night").alias("Night")).toPandas().iloc[0]
borrow_time_sum.plot.pie(autopct='%1.1f%%', figsize=(8, 8), colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"])
plt.title("Borrow Time Period Distribution")
plt.ylabel("")
plt.show()

# 3. 散点图（聚类结果）
clustered_data_pandas = clustered_data.select("reader_id", "entry_count", "borrow_count", "seat_usage_count", "cluster").toPandas()
plt.figure(figsize=(10, 6))
sns.scatterplot(data=clustered_data_pandas, x="entry_count", y="borrow_count", hue="cluster", palette="Set2", s=100)
plt.title("User Clustering Based on Entry and Borrow Count")
plt.xlabel("Entry Count")
plt.ylabel("Borrow Count")
plt.legend(title="Cluster")
plt.show()
