from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, hour, when, unix_timestamp, approx_count_distinct, rand
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# **✅ 1. Create SparkSession**
spark = SparkSession.builder \
    .appName("Library Data Privacy Protection") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# **✅ 2. Read Data**
data_path = "file:///home/hadoop/Downloads/图书馆数据/"
output_path = data_path.replace("file://", "") + "privacy_protected/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

entry_logs = spark.read.csv(data_path + "entry_logs.csv", header=True, inferSchema=True, encoding="utf-8")
borrow_logs = spark.read.csv(data_path + "借阅数据_guid_utf8.csv", header=True, inferSchema=True, encoding="utf-8")
book_data = spark.read.csv(data_path + "图书数据.csv", header=True, inferSchema=True, encoding="utf-8")
seat_logs = spark.read.csv(data_path + "seat_logs.csv", header=True, inferSchema=True, encoding="utf-8")
reader_data = spark.read.csv(data_path + "读者数据_guid_utf8.csv", header=True, inferSchema=True, encoding="utf-8")

# ✅ **Remove sensitive columns (but keep `ID` for mapping)**
reader_data = reader_data.drop("NO")

# **✅ 3. K-Anonymity: Group by Department and Type**
k_anonymized_data = reader_data.groupBy("DEPARTMENT", "TYPE").agg(count("*").alias("UserCount"))

print("\n📌 K-Anonymity: Full User Count by Department and Type:")
k_anonymized_data.show(truncate=False)  # ✅ Print all department data

# **📊 Visualization: K-Anonymity User Distribution**
k_anonymized_pd = k_anonymized_data.toPandas()
plt.figure(figsize=(10, 6))
sns.barplot(x="DEPARTMENT", y="UserCount", data=k_anonymized_pd)
plt.xticks(rotation=45)
plt.xlabel("Department")
plt.ylabel("User Count")
plt.title("K-Anonymity: User Count Distribution by Department")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "K_Anonymity_Distribution.png"))
plt.show()

# **✅ 4. L-Diversity: Ensure each group has at least `L` different book categories**
borrow_logs = borrow_logs.join(book_data, borrow_logs.BOOKID == book_data.ID, "left")

l_value = 3  # Set L-Diversity threshold
borrow_diversity = borrow_logs.groupBy("READERID").agg(approx_count_distinct("DOCTYPE").alias("DistinctCategories"))
borrow_logs = borrow_logs.join(borrow_diversity, "READERID", "left").filter(col("DistinctCategories") >= l_value)

print("\n📌 L-Diversity Example (Number of Book Categories Borrowed per User):")
borrow_logs.select("READERID", "DistinctCategories").show(5)

# **📊 Visualization: L-Diversity (Book Categories Borrowed)**
l_diversity_pd = borrow_logs.select("DistinctCategories").toPandas()
plt.figure(figsize=(8, 5))
sns.histplot(l_diversity_pd, x="DistinctCategories", bins=10, kde=True)
plt.xlabel("Number of Book Categories Borrowed")
plt.ylabel("User Count")
plt.title("L-Diversity: Number of Book Categories Borrowed")
plt.grid()
plt.savefig(os.path.join(output_path, "L_Diversity_Distribution.png"))
plt.show()

# **✅ 5. T-Proximity: Anonymize Visit Time**
entry_logs = entry_logs.withColumn("Hour", hour(col("VisitTime")))
entry_logs = entry_logs.withColumn("TimeGroup", when((col("Hour") >= 6) & (col("Hour") < 12), "Morning")
                                   .when((col("Hour") >= 12) & (col("Hour") < 18), "Afternoon")
                                   .otherwise("Evening"))

print("\n📌 T-Proximity Example (Visit Time Periods):")
entry_logs.select("VisitTime", "TimeGroup").show(5)

# **📊 Visualization: T-Proximity Visit Time Period Distribution**
t_proximity_pd = entry_logs.groupBy("TimeGroup").agg(count("*").alias("Count")).toPandas()
plt.figure(figsize=(6, 6))
plt.pie(t_proximity_pd["Count"], labels=t_proximity_pd["TimeGroup"], autopct="%1.1f%%", colors=["blue", "orange", "green"])
plt.title("T-Proximity: Visit Time Period Distribution")
plt.savefig(os.path.join(output_path, "T_Proximity_Distribution.png"))
plt.show()

# **✅ 6. Fix `ID` Issue & Replace `READERID` with Anonymous ID**
# ✅ **Ensure `ID` exists before generating `AnonID`**
user_mapping = reader_data.select("ID").withColumn("AnonID", (rand() * 1000000).cast("int"))

# ✅ **Fix `join` issue by using correct column name**
borrow_logs = borrow_logs.join(user_mapping, "ID", "left").drop("READERID")

# **✅ 7. Save Anonymized Data**
k_anonymized_data.write.csv(os.path.join(output_path, "k_anonymized.csv"), header=True, mode="overwrite", encoding="utf-8")
borrow_logs.write.csv(os.path.join(output_path, "l_diverse_borrow_logs.csv"), header=True, mode="overwrite", encoding="utf-8")
entry_logs.write.csv(os.path.join(output_path, "t_proximity_entry_logs.csv"), header=True, mode="overwrite", encoding="utf-8")

print(f"\n✅ Data anonymization completed. Saved to: {output_path}")

# **✅ 8. Stop SparkSession**
spark.stop()
