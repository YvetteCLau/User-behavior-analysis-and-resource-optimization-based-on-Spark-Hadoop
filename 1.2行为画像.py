from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, hour, month, avg, when, unix_timestamp

# 创建 SparkSession
spark = SparkSession.builder \
    .appName("User Behavior Profiling") \
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

# 1. **常驻入馆时间段**
entry_logs = entry_logs.withColumn("Hour", hour(col("VisitTime")))

entry_logs = entry_logs.withColumn("TimePeriod", 
    when((col("Hour") >= 6) & (col("Hour") < 12), "Morning")
    .when((col("Hour") >= 12) & (col("Hour") < 18), "Afternoon")
    .otherwise("Evening"))

user_visit_profile = entry_logs.groupBy("ID", "TimePeriod") \
                               .agg(count("*").alias("VisitCount"))

# 2. **书籍偏好**
borrow_book_df = borrow_logs.join(book_data, borrow_logs.BOOKID == book_data.ID, "left")

user_book_preference = borrow_book_df.groupBy("READERID", "DOCTYPE", "LANGUAGE") \
                                     .agg(count("*").alias("BorrowCount"))

# 3. **座位偏好**
seat_logs = seat_logs.withColumn("UsageTime", 
    (unix_timestamp(col("LeaveSeatTime")) - unix_timestamp(col("SelectSeatTime"))) / 3600)

user_seat_profile = seat_logs.groupBy("ID") \
                             .agg(avg("UsageTime").alias("AvgUsageTime"),
                                  count("*").alias("UsageFrequency"))

reading_room_preference = seat_logs.groupBy("ID", "ReadingRoomNo") \
                                   .agg(count("*").alias("RoomUsageCount"))

# 4. **用户群体特征**
user_group_profile = reader_data.select("ID", "GENDER", "DEPARTMENT", "TYPE")

# 5. **多模态数据融合（全景用户画像）**
user_profile = user_group_profile \
    .join(user_visit_profile, "ID", "left") \
    .join(user_book_preference, user_group_profile["ID"] == user_book_preference["READERID"], "left") \
    .join(user_seat_profile, "ID", "left") \
    .join(reading_room_preference, "ID", "left") \
    .drop("READERID")

# 6. **展示结果**
print("用户入馆时间偏好：")
user_visit_profile.show()

print("用户书籍偏好：")
user_book_preference.show()

print("用户座位偏好：")
user_seat_profile.show()

print("用户群体特征：")
user_group_profile.show()

print("全景用户画像：")
user_profile.show()

# 7. **保存结果**
output_path = data_path + "user_profiles/"
user_profile.write.csv(output_path + "user_profile.csv", header=True, mode="overwrite", encoding="utf-8")

print(f"用户画像数据已保存至：{output_path}")

# 8. 结束 SparkSession
spark.stop()
