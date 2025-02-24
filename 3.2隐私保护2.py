import os  # ✅ 确保 `os` 被正确导入
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, hour, when, approx_count_distinct, rand

# **✅ 1. 创建 SparkSession**
spark = SparkSession.builder \
    .appName("Library Data Privacy Protection") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# **✅ 2. 读取数据**
data_path = "file:///home/hadoop/Downloads/图书馆数据/"
output_dir = "file:///home/hadoop/Downloads/privacy_protected/"  # ✅ 确保是 `file://` 本地路径

# **✅ 确保 `privacy_protected` 文件夹存在**
os.makedirs(output_dir.replace("file://", ""), exist_ok=True)  # `os.makedirs()` 不能使用 `file://`

entry_logs = spark.read.csv(data_path + "entry_logs.csv", header=True, inferSchema=True, encoding="utf-8")
borrow_logs = spark.read.csv(data_path + "借阅数据_guid_utf8.csv", header=True, inferSchema=True, encoding="utf-8")
book_data = spark.read.csv(data_path + "图书数据.csv", header=True, inferSchema=True, encoding="utf-8")
seat_logs = spark.read.csv(data_path + "seat_logs.csv", header=True, inferSchema=True, encoding="utf-8")
reader_data = spark.read.csv(data_path + "读者数据_guid_utf8.csv", header=True, inferSchema=True, encoding="utf-8")

# ✅ **修复 `COLUMN_ALREADY_EXISTS` 问题**
if "NO" in book_data.columns:
    book_data = book_data.drop("NO")  # 移除 `book_data` 中的 `NO` 列，避免 `join` 冲突

# **✅ 3. L-多样性：确保每个用户至少借阅 `L` 种不同类别的书籍**
borrow_logs = borrow_logs.join(book_data, borrow_logs.BOOKID == book_data.ID, "left")

l_value = 3  # 设置 L-多样性阈值
borrow_diversity = borrow_logs.groupBy("READERID").agg(approx_count_distinct("DOCTYPE").alias("DistinctCategories"))
borrow_logs = borrow_logs.join(borrow_diversity, "READERID", "left").filter(col("DistinctCategories") >= l_value)

# **✅ 4. T-接近性：模糊访问时间**
entry_logs = entry_logs.withColumn("Hour", hour(col("VisitTime")))
entry_logs = entry_logs.withColumn("TimeGroup", when((col("Hour") >= 6) & (col("Hour") < 12), "Morning")
                                   .when((col("Hour") >= 12) & (col("Hour") < 18), "Afternoon")
                                   .otherwise("Evening"))

# **✅ 5. 生成匿名 ID 并替换 `READERID`**
user_mapping = reader_data.select("ID").withColumn("AnonID", (rand() * 1000000).cast("int"))
borrow_logs = borrow_logs.join(user_mapping, "ID", "left").drop("READERID")

# **✅ 6. 存储匿名数据到 `privacy_protected/`，确保 CSV 只有一个文件**
borrow_logs.coalesce(1).write.csv(output_dir + "l_diverse_borrow_logs.csv", header=True, mode="overwrite", encoding="utf-8")
entry_logs.coalesce(1).write.csv(output_dir + "t_proximity_entry_logs.csv", header=True, mode="overwrite", encoding="utf-8")

# **✅ 7. 关闭 Spark**
spark.stop()
