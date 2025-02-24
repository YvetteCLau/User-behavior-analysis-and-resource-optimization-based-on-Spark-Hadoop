from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder \
    .appName("Library Data Integration") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# 数据文件路径
data_path = "file:///home/hadoop/Downloads/图书馆数据/"

# 读取数据
reader_data = spark.read.csv(data_path + "读者数据_guid_utf8.csv", header=True, inferSchema=True, encoding="utf-8")
entry_logs = spark.read.csv(data_path + "entry_logs.csv", header=True, inferSchema=True, encoding="utf-8")
borrow_logs = spark.read.csv(data_path + "借阅数据_guid_utf8.csv", header=True, inferSchema=True, encoding="utf-8")
seat_logs = spark.read.csv(data_path + "seat_logs.csv", header=True, inferSchema=True, encoding="utf-8")

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

# 整合读者信息
user_data = reader_data \
    .join(entry_logs, on="ID", how="left") \
    .join(borrow_logs, reader_data["ID"] == borrow_logs["READERID"], how="left") \
    .join(seat_logs, on="ID", how="left")

# 打印整合后数据的预览
print("\n整合后的用户数据预览：")
user_data.show(5)

# 保存整合后的数据到 CSV 文件
output_path = data_path + "user_data"
user_data.write.csv(output_path, header=True, mode="overwrite", encoding="utf-8")

print(f"整合后的用户数据已保存到：{output_path}")
