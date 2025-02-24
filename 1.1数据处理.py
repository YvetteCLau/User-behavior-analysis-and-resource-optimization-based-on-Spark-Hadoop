from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, hour, month, avg, unix_timestamp

# 创建 SparkSession
spark = SparkSession.builder \
    .appName("Library Data Processing") \
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
book_data = spark.read.csv(data_path + "图书数据.csv", header=True, inferSchema=True, encoding="utf-8")

# **移除 reader_data 表中的 NO 列**
reader_data = reader_data.drop("NO")

# 1. 处理入馆记录数据
## 计算入馆时间频率（按小时统计）
visit_freq = entry_logs.withColumn("Hour", hour(col("VisitTime"))) \
                       .groupBy("Hour") \
                       .agg(count("*").alias("VisitCount")) \
                       .orderBy("Hour")

## 计算入馆地点偏好（统计每个地点的访问次数）
location_preference = entry_logs.groupBy("Location") \
                                .agg(count("*").alias("VisitCount")) \
                                .orderBy(col("VisitCount").desc())

# 2. 处理借阅数据
## 计算借阅时间分布（按月份统计）
borrow_time_distribution = borrow_logs.withColumn("Month", month(col("LENDDATE"))) \
                                      .groupBy("Month") \
                                      .agg(count("*").alias("BorrowCount")) \
                                      .orderBy("Month")

## 关联借阅数据和图书数据，计算偏好的书籍类别和语言
borrow_book_df = borrow_logs.join(book_data, borrow_logs.BOOKID == book_data.ID, "left")

## 统计最受欢迎的书籍类别（DOCTYLE）
book_type_preference = borrow_book_df.groupBy("DOCTYPE") \
                                     .agg(count("*").alias("BorrowCount")) \
                                     .orderBy(col("BorrowCount").desc())

## 统计最受欢迎的书籍语言（LANGUAGE）
language_preference = borrow_book_df.groupBy("LANGUAGE") \
                                    .agg(count("*").alias("BorrowCount")) \
                                    .orderBy(col("BorrowCount").desc())

# 3. 处理座位记录
## 计算用户使用座位的平均时长 (单位：小时)
seat_usage_time = seat_logs.withColumn("UsageTime", 
                                       (unix_timestamp(col("LeaveSeatTime")) - unix_timestamp(col("SelectSeatTime"))) / 3600) \
                           .groupBy("ID") \
                           .agg(avg("UsageTime").alias("AvgUsageTime")) \
                           .orderBy(col("AvgUsageTime").desc())

## 计算偏好的阅览室
reading_room_preference = seat_logs.groupBy("ReadingRoomNo") \
                                   .agg(count("*").alias("UsageCount")) \
                                   .orderBy(col("UsageCount").desc())

# 4. 显示结果
print("入馆时间频率（按小时）：")
visit_freq.show()

print("入馆地点偏好：")
location_preference.show()

print("借阅时间分布（按月份）：")
borrow_time_distribution.show()

print("偏好书籍类别：")
book_type_preference.show()

print("偏好书籍语言：")
language_preference.show()

print("用户使用座位的平均时长（小时）：")
seat_usage_time.show()

print("偏好的阅览室：")
reading_room_preference.show()

# 5. 保存结果到 CSV
output_path = data_path + "processed_data/"
visit_freq.write.csv(output_path + "visit_freq.csv", header=True, mode="overwrite", encoding="utf-8")
location_preference.write.csv(output_path + "location_preference.csv", header=True, mode="overwrite", encoding="utf-8")
borrow_time_distribution.write.csv(output_path + "borrow_time_distribution.csv", header=True, mode="overwrite", encoding="utf-8")
book_type_preference.write.csv(output_path + "book_type_preference.csv", header=True, mode="overwrite", encoding="utf-8")
language_preference.write.csv(output_path + "language_preference.csv", header=True, mode="overwrite", encoding="utf-8")
seat_usage_time.write.csv(output_path + "seat_usage_time.csv", header=True, mode="overwrite", encoding="utf-8")
reading_room_preference.write.csv(output_path + "reading_room_preference.csv", header=True, mode="overwrite", encoding="utf-8")

print(f"所有处理结果已保存到：{output_path}")

# 6. 结束 SparkSession
spark.stop()
