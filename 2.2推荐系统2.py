from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, udf, explode, month, year
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, StopWordsRemover, StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.clustering import LDA
from pyspark.sql.types import StringType, IntegerType, FloatType
from pyspark.ml.linalg import SparseVector, DenseVector, VectorUDT
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

# **5️⃣ NLP 主题挖掘**
def jieba_tokenizer(text):
    if text is None:
        return ""
    return " ".join(jieba.cut(text))

tokenizer_udf = udf(jieba_tokenizer, StringType())  
book_data = book_data.withColumn("Tokens", tokenizer_udf(col("TITLE")))  

# **6️⃣ 处理文本特征**
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

# **7️⃣ LDA 主题建模**
lda = LDA(k=5, featuresCol="TFIDF", seed=42)
lda_model = lda.fit(tfidf_data)
topics = lda_model.describeTopics(5)

# **8️⃣ ALS 推荐系统**
reader_indexer = StringIndexer(inputCol="READERID", outputCol="READERID_Index")
book_indexer = StringIndexer(inputCol="BOOKID", outputCol="BOOKID_Index")

borrow_logs = reader_indexer.fit(borrow_logs).transform(borrow_logs)
borrow_logs = book_indexer.fit(borrow_logs).transform(borrow_logs)

als_data = borrow_logs.groupBy("READERID_Index", "BOOKID_Index").agg(count("*").alias("BorrowCount"))
als_data = als_data.dropna(subset=["READERID_Index", "BOOKID_Index", "BorrowCount"])

if als_data.count() == 0:
    raise ValueError("❌ ALS 训练数据为空，请检查 borrow_logs 是否有数据！")

als = ALS(userCol="READERID_Index", itemCol="BOOKID_Index", ratingCol="BorrowCount",
          coldStartStrategy="drop", implicitPrefs=True)
als_model = als.fit(als_data)

user_recommendations = als_model.recommendForAllUsers(5)

# ✅ **修复 `recommendations` 结构体数组问题**
user_recommendations = user_recommendations.withColumn("recommendation", explode(col("recommendations")))
user_recommendations = user_recommendations.select("READERID_Index", 
                                                   col("recommendation.BOOKID_Index").alias("BOOKID_Index"), 
                                                   col("recommendation.rating").alias("Rating"))

# **9️⃣ 计算书籍相似度**
def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

cosine_udf = udf(cosine_similarity, FloatType())

# ✅ **优化计算逻辑：仅计算 Top 100 热门书籍的相似度**
top_books = borrow_logs.groupBy("BOOKID").agg(count("*").alias("BorrowCount")).orderBy(col("BorrowCount").desc()).limit(100)
top_books = top_books.join(tfidf_data, top_books.BOOKID == tfidf_data.ID, "inner").select("ID", "TFIDF")

book_similarity = top_books.alias("a").crossJoin(top_books.alias("b")) \
    .withColumn("Similarity", cosine_udf(col("a.TFIDF"), col("b.TFIDF"))) \
    .filter(col("a.ID") != col("b.ID")) \
    .select(col("a.ID").alias("Book1"), col("b.ID").alias("Book2"), col("Similarity")) \
    .orderBy(col("Similarity").desc()).limit(50)

# **📌 结果存储**
output_path = "file:///home/hadoop/Downloads/推荐系统结果/"

user_recommendations.write.csv(output_path + "user_recommendations.csv", header=True, mode="overwrite")
book_similarity.write.csv(output_path + "book_similarity.csv", header=True, mode="overwrite")
topics.write.csv(output_path + "lda_topics.csv", header=True, mode="overwrite")
als_data.write.csv(output_path + "als_data.csv", header=True, mode="overwrite")

# **📌 关闭 Spark**
spark.stop()
