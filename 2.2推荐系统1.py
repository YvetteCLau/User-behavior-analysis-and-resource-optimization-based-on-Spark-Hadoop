from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, udf, collect_set, month, year
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, StopWordsRemover, StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.clustering import LDA
from pyspark.sql.types import StringType, IntegerType, FloatType
import jieba
import numpy as np

# **1️⃣ 创建 SparkSession**
spark = SparkSession.builder \
    .appName("Library User Interest & Recommendation") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# **2️⃣ 读取数据**
data_path = "file:///home/hadoop/Downloads/图书馆数据/"

reader_data = spark.read.csv(data_path + "读者数据_guid_utf8.csv", header=True, inferSchema=True)
borrow_logs = spark.read.csv(data_path + "借阅数据_guid_utf8.csv", header=True, inferSchema=True)
book_data = spark.read.csv(data_path + "图书数据.csv", header=True, inferSchema=True)

# ✅ **检查 `borrow_logs` 是否为空**
if borrow_logs.count() == 0:
    raise ValueError("❌ borrow_logs 数据为空，请检查数据文件是否正确加载！")

# **3️⃣ 数据清理**
reader_data = reader_data.select("ID", "GENDER", "ENROLLYEAR", "TYPE", "DEPARTMENT")
borrow_logs = borrow_logs.select("READERID", "BOOKID", "LENDDATE")
book_data = book_data.select("ID", "TITLE", "AUTHOR", "PUBLISHER", "YEAR", "CALLNO", "LANGUAGE", "DOCTYPE")

# **4️⃣ 处理时间信息**
borrow_logs = borrow_logs.withColumn("Year", year(col("LENDDATE"))) \
                         .withColumn("Month", month(col("LENDDATE")))

# **5️⃣ 用户画像分析**
user_borrow_count = borrow_logs.groupBy("READERID").agg(count("*").alias("BorrowCount"))
user_category = borrow_logs.join(book_data, borrow_logs.BOOKID == book_data.ID, "left") \
                           .groupBy("READERID").agg(collect_set("DOCTYPE").alias("FavCategories"))
user_profile = user_borrow_count.join(user_category, "READERID", "left")

# **6️⃣ NLP 主题挖掘**
def jieba_tokenizer(text):
    if text is None:
        return ""
    return " ".join(jieba.cut(text))

tokenizer_udf = udf(jieba_tokenizer, StringType())  
book_data = book_data.withColumn("Tokens", tokenizer_udf(col("TITLE")))  

# **7️⃣ 使用 Tokenizer**
tokenizer = Tokenizer(inputCol="Tokens", outputCol="Words")  
words_data = tokenizer.transform(book_data)

remover = StopWordsRemover(inputCol="Words", outputCol="FilteredWords")
filtered_data = remover.transform(words_data)

vectorizer = CountVectorizer(inputCol="FilteredWords", outputCol="Features")
cv_model = vectorizer.fit(filtered_data)
vectorized_data = cv_model.transform(filtered_data)

idf = IDF(inputCol="Features", outputCol="TFIDF")
idf_model = idf.fit(vectorized_data)
tfidf_data = idf_model.transform(vectorized_data)

# **8️⃣ LDA 主题建模**
lda = LDA(k=5, featuresCol="TFIDF", seed=42)
lda_model = lda.fit(tfidf_data)
topics = lda_model.describeTopics(5)

# **9️⃣ 训练 ALS 推荐系统**
# ✅ **将 `READERID` 和 `BOOKID` 转换为索引编号**
reader_indexer = StringIndexer(inputCol="READERID", outputCol="READERID_Index")
book_indexer = StringIndexer(inputCol="BOOKID", outputCol="BOOKID_Index")

borrow_logs = reader_indexer.fit(borrow_logs).transform(borrow_logs)
borrow_logs = book_indexer.fit(borrow_logs).transform(borrow_logs)

# ✅ **构建 ALS 训练数据**
als_data = borrow_logs.groupBy("READERID_Index", "BOOKID_Index").agg(count("*").alias("BorrowCount"))

# ✅ **过滤掉 NULL 值**
als_data = als_data.dropna(subset=["READERID_Index", "BOOKID_Index", "BorrowCount"])

# ✅ **检查 ALS 训练数据是否为空**
if als_data.count() == 0:
    print("⚠️ ALS 训练数据为空，尝试打印 borrow_logs 的前 5 行：")
    borrow_logs.show(5, truncate=False)
    raise ValueError("❌ ALS 训练数据为空，请检查 borrow_logs 是否有数据！")

# ✅ **训练 ALS 模型**
als = ALS(userCol="READERID_Index", itemCol="BOOKID_Index", ratingCol="BorrowCount",
          coldStartStrategy="drop", implicitPrefs=True)
als_model = als.fit(als_data)

# **🔟 生成推荐**
user_recommendations = als_model.recommendForAllUsers(5)

# **📌 计算基于内容的推荐**
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

cosine_udf = udf(cosine_similarity, FloatType())

book_similarity = tfidf_data.alias("a").crossJoin(tfidf_data.alias("b")) \
    .withColumn("Similarity", cosine_udf(col("a.TFIDF"), col("b.TFIDF"))) \
    .filter(col("a.ID") != col("b.ID")) \
    .select(col("a.ID").alias("Book1"), col("b.ID").alias("Book2"), col("Similarity")) \
    .orderBy(col("Similarity").desc())

# **📌 动态推荐（时间调整）**
borrow_trends = borrow_logs.groupBy("Year", "Month", "BOOKID").agg(count("*").alias("BorrowCount"))
borrow_trends = borrow_trends.orderBy(col("Year").desc(), col("Month").desc())

# **📌 展示结果**
print("📌 用户画像（Top 5）：")
user_profile.show(5, truncate=False)

print("📌 书籍主题分析（Top 5）：")
topics.show(5, truncate=False)

print("📌 协同过滤推荐（Top 5 用户推荐）：")
user_recommendations.show(5, truncate=False)

print("📌 书籍相似度（Top 5 相似书籍）：")
book_similarity.show(5, truncate=False)

print("📌 借阅趋势分析（Top 5 最新趋势）：")
borrow_trends.show(5, truncate=False)

# **📌 保存分析结果**
output_path = "file:///home/hadoop/Downloads/推荐系统结果/"

user_profile.write.csv(output_path + "user_profiles.csv", header=True, mode="overwrite")
user_recommendations.write.csv(output_path + "user_recommendations.csv", header=True, mode="overwrite")
book_similarity.write.csv(output_path + "book_similarity.csv", header=True, mode="overwrite")
borrow_trends.write.csv(output_path + "borrow_trends.csv", header=True, mode="overwrite")

# **📌 关闭 Spark**
spark.stop()
