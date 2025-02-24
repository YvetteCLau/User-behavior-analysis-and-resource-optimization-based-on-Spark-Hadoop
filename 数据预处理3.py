from pyspark.sql import SparkSession
# 创建 SparkSession
spark = SparkSession.builder \
    .appName("Library Data Integration") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.default.parallelism", "200") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .getOrCreate()

# 数据文件路径
data_path = "file:///home/hadoop/Downloads/图书馆数据/"

# 读取数据
reader_data = spark.read.csv(data_path + "读者数据_guid_utf8.csv", header=True, inferSchema=True, encoding="utf-8")
entry_logs = spark.read.csv(data_path + "entry_logs.csv", header=True, inferSchema=True, encoding="utf-8")
borrow_logs = spark.read.csv(data_path + "借阅数据_guid_utf8.csv", header=True, inferSchema=True, encoding="utf-8")
seat_logs = spark.read.csv(data_path + "seat_logs.csv", header=True, inferSchema=True, encoding="utf-8")

# 将数据保存为 Parquet 格式以提高性能
reader_data.write.parquet(data_path + "读者数据_guid_utf8.parquet", mode="overwrite")
entry_logs.write.parquet(data_path + "entry_logs.parquet", mode="overwrite")
borrow_logs.write.parquet(data_path + "借阅数据_guid_utf8.parquet", mode="overwrite")
seat_logs.write.parquet(data_path + "seat_logs.parquet", mode="overwrite")

# 重新读取 Parquet 格式数据
reader_data = spark.read.parquet(data_path + "读者数据_guid_utf8.parquet")
entry_logs = spark.read.parquet(data_path + "entry_logs.parquet")
borrow_logs = spark.read.parquet(data_path + "借阅数据_guid_utf8.parquet")
seat_logs = spark.read.parquet(data_path + "seat_logs.parquet")

# 打印每个数据表的架构
print("读者数据表架构：")
reader_data.printSchema()

print("\n门禁数据表架构：")
entry_logs.printSchema()

print("\n借阅数据表架构：")
borrow_logs.printSchema()

print("\n座位数据表架构：")
seat_logs.printSchema()

# **移除 reader_data 表中的 NO 列**
reader_data = reader_data.drop("NO")

# 重命名列，避免歧义
reader_data = reader_data.withColumnRenamed("ID", "reader_id")
entry_logs = entry_logs.withColumnRenamed("ID", "entry_id")
borrow_logs = borrow_logs.withColumnRenamed("READERID", "borrow_reader_id")
seat_logs = seat_logs.withColumnRenamed("ID", "seat_id")

# Join 操作：与 entry_logs 合并
reader_entry_data = reader_data.join(
    entry_logs,
    reader_data["reader_id"] == entry_logs["entry_id"],
    how="left"
)

# Join 操作：与 borrow_logs 合并
reader_entry_borrow_data = reader_entry_data.join(
    borrow_logs,
    reader_entry_data["reader_id"] == borrow_logs["borrow_reader_id"],
    how="left"
)

# Join 操作：与 seat_logs 合并
user_data = reader_entry_borrow_data.join(
    seat_logs,
    reader_entry_borrow_data["reader_id"] == seat_logs["seat_id"],
    how="left"
)

# 打印整合后的数据
user_data.show(5)
