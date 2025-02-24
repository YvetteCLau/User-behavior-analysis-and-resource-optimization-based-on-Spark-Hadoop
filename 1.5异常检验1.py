from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, hour, dayofweek, to_date, max
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number  # **导入 row_number()**

# **1. 创建 SparkSession**
spark = SparkSession.builder.appName("Library Access Analysis").getOrCreate()

# **2. 读取门禁数据**
data_path = "file:///home/hadoop/Downloads/图书馆数据/"
entry_logs = spark.read.csv(data_path + "entry_logs.csv", header=True, inferSchema=True)

# **3. 处理入馆记录**
entry_logs = entry_logs.withColumn("Hour", hour(col("VisitTime"))) \
                       .withColumn("Weekday", dayofweek(col("VisitTime"))) \
                       .withColumn("Date", to_date(col("VisitTime")))

# **4. 统计访问时间分布**
hourly_visit_dist = entry_logs.groupBy("Hour").agg(count("*").alias("VisitCount")).orderBy(col("VisitCount").desc()).limit(10)

# **5. 统计访问最多的地点**
location_preference = entry_logs.groupBy("Location").agg(count("*").alias("VisitCount")).orderBy(col("VisitCount").desc()).limit(10)

# **6. 计算活跃用户**
last_visit = entry_logs.groupBy("ID").agg(max("Date").alias("LastVisit"))
window_spec = Window.orderBy(col("LastVisit").desc())
last_visit = last_visit.withColumn("Rank", row_number().over(window_spec))

active_users = last_visit.filter(col("Rank") <= 10)  # 最近访问排名前 10

# **7. 展示结果**
print("\nTop 10 - 按小时统计访问量：")
hourly_visit_dist.show()

print("\n📌 Top 10 - 访问最频繁的地点：")
location_preference.show()

print("\n Top 10 - 活跃用户（最近访问）：")
active_users.show()

# **8. 存储分析结果**
output_path = "file:///home/hadoop/Downloads/分析结果/"

hourly_visit_dist.write.csv(output_path + "hourly_visit_dist.csv", header=True, mode="overwrite")
location_preference.write.csv(output_path + "location_preference.csv", header=True, mode="overwrite")
active_users.write.csv(output_path + "active_users.csv", header=True, mode="overwrite")

# **9. 关闭 Spark**
spark.stop()
