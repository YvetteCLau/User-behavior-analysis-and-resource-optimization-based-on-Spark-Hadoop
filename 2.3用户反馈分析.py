from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, sum as spark_sum, datediff, current_date, unix_timestamp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# **1️⃣ 创建 SparkSession**
spark = SparkSession.builder \
    .appName("Library User Satisfaction Analysis") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# **2️⃣ 读取数据**
data_path = "file:///home/hadoop/Downloads/图书馆数据/"

reader_data = spark.read.csv(data_path + "读者数据_guid_utf8.csv", header=True, inferSchema=True)
borrow_logs = spark.read.csv(data_path + "借阅数据_guid_utf8.csv", header=True, inferSchema=True)
book_data = spark.read.csv(data_path + "图书数据.csv", header=True, inferSchema=True)
seat_logs = spark.read.csv(data_path + "seat_logs.csv", header=True, inferSchema=True)  # ✅ 修正文件名

# ✅ **检查 `borrow_logs` 是否为空**
if borrow_logs.count() == 0:
    raise ValueError("❌ borrow_logs 数据为空，请检查数据文件是否正确加载！")

# **3️⃣ 借阅续借次数分析**
borrow_logs = borrow_logs.withColumn("Renewed", when(col("RENEWCOUNTS") > 0, 1).otherwise(0))  # ✅ 修正列名
user_satisfaction = borrow_logs.groupBy("READERID") \
                               .agg(spark_sum("Renewed").alias("TotalRenewals"),
                                    count("*").alias("TotalBorrows")) \
                               .withColumn("RenewalRate", col("TotalRenewals") / col("TotalBorrows"))

# **4️⃣ 计算未归还图书**
borrow_logs = borrow_logs.withColumn("OverdueDays", datediff(current_date(), col("RETURNDATE")))  # ✅ 修正列名
overdue_books = borrow_logs.filter(col("OverdueDays") > 0) \
                           .groupBy("BOOKID") \
                           .agg(count("*").alias("OverdueCount")) \
                           .orderBy(col("OverdueCount").desc())

# **5️⃣ 座位使用分析**
seat_logs = seat_logs.withColumn("UsageDuration", 
                                 (unix_timestamp(col("LeaveSeatTime")) - unix_timestamp(col("SelectSeatTime"))) / 60)  # ✅ 计算分钟数

seat_issues = seat_logs.filter(col("UsageDuration") < 30) \
                       .groupBy("SeatNo") \
                       .agg(count("*").alias("ShortUsageCount")) \
                       .orderBy(col("ShortUsageCount").desc())

# **📌 结果存储**
output_path = "file:///home/hadoop/Downloads/满意度分析结果/"

user_satisfaction.write.csv(output_path + "user_satisfaction.csv", header=True, mode="overwrite")
overdue_books.write.csv(output_path + "overdue_books.csv", header=True, mode="overwrite")
seat_issues.write.csv(output_path + "seat_issues.csv", header=True, mode="overwrite")

# **6️⃣ Top 10 结果转换为 Pandas DataFrame**
top_users = user_satisfaction.orderBy(col("TotalRenewals").desc()).limit(10).toPandas()
top_books = overdue_books.limit(10).toPandas()
top_seats = seat_issues.limit(10).toPandas()

# **✅ 直接打印数据**
print("\n📌 借阅续借次数最多的前 10 名用户：")
print(top_users)

print("\n📌 未归还次数最多的前 10 本书：")
print(top_books)

print("\n📌 频繁短时使用的前 10 个座位：")
print(top_seats)

# **7️⃣ 可视化数据**
sns.set(style="whitegrid")

# ✅ **支持中文**
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于 Windows/Linux
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

plt.figure(figsize=(15, 5))

# 📌 借阅续借次数最多的前 10 名用户
plt.subplot(1, 3, 1)
sns.barplot(x="TotalRenewals", y="READERID", data=top_users, palette="Blues_r")
plt.xlabel("续借次数")
plt.ylabel("用户 ID")
plt.title("借阅续借次数最多的前 10 名用户")

# 📌 未归还次数最多的前 10 本书
plt.subplot(1, 3, 2)
sns.barplot(x="OverdueCount", y="BOOKID", data=top_books, palette="Reds_r")
plt.xlabel("未归还次数")
plt.ylabel("图书 ID")
plt.title("未归还次数最多的前 10 本书")

# 📌 离座时间短的前 10 个座位
plt.subplot(1, 3, 3)
sns.barplot(x="ShortUsageCount", y="SeatNo", data=top_seats, palette="Greens_r")
plt.xlabel("短时间使用次数")
plt.ylabel("座位号")
plt.title("频繁短时使用的前 10 个座位")

plt.tight_layout()
plt.show()

# **📌 关闭 Spark**
spark.stop()
