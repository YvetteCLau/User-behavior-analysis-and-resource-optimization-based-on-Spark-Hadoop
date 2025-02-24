from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, regexp_extract, lower, trim, udf, when, concat_ws
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, StopWordsRemover
from pyspark.ml.clustering import LDA
from pyspark.sql.types import StringType
import os
import jieba

# **1. 创建 SparkSession**
spark = SparkSession.builder \
    .appName("Library Homepage Analysis") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# **2. 读取图书馆主页文章数据**
articles_path = "file:///home/hadoop/Downloads/图书馆数据/图书馆主页数据/图书馆主页网站文章.csv"

if not os.path.exists(articles_path.replace("file://", "")):
    raise FileNotFoundError(f"❌ 文章数据文件未找到: {articles_path}")

articles = spark.read.csv(articles_path, header=True, inferSchema=True)

# **3. 处理列名**
articles = articles.toDF(*[c.strip() for c in articles.columns])

# **4. 统一列名**
articles = articles.select(
    col("栏目").alias("Column"),
    col("标题").alias("Title"),
    col("链接").alias("URL"),
    col("发布日期").alias("PublishDate"),
    col("访问量").alias("Views"),
    col("文章内容").alias("Content")
)

# **5. 处理文章内容**
articles = articles.withColumn("Content", when(col("Content").isNull(), "").otherwise(col("Content")))
articles = articles.withColumn("Content", trim(col("Content")))
articles = articles.withColumn("LowerContent", lower(col("Content")))

# **6. 使用 NLP 提取文章关键词**
def jieba_tokenizer(text):
    if text is None:
        return ""
    return " ".join(jieba.cut(text))

tokenizer_udf = udf(jieba_tokenizer, StringType())

articles = articles.withColumn("Tokens", tokenizer_udf(col("LowerContent")))

tokenizer = Tokenizer(inputCol="Tokens", outputCol="Words")
words_data = tokenizer.transform(articles)

remover = StopWordsRemover(inputCol="Words", outputCol="FilteredWords")
filtered_data = remover.transform(words_data)

vectorizer = CountVectorizer(inputCol="FilteredWords", outputCol="Features")
cv_model = vectorizer.fit(filtered_data)
vectorized_data = cv_model.transform(filtered_data)

idf = IDF(inputCol="Features", outputCol="TFIDF")
idf_model = idf.fit(vectorized_data)
tfidf_data = idf_model.transform(vectorized_data)

# **7. 使用 LDA 进行主题建模**
lda = LDA(k=5, featuresCol="TFIDF", seed=42)
lda_model = lda.fit(tfidf_data)
topics = lda_model.describeTopics(5)

# **8. 修正 LDA 结果，转换数组列**
topics = topics.withColumn("termIndices", concat_ws(",", col("termIndices")))
topics = topics.withColumn("termWeights", concat_ws(",", col("termWeights")))

# **9. 读取访问日志**
logs_path = "file:///home/hadoop/Downloads/图书馆数据/图书馆主页数据/"

if not os.path.exists(logs_path.replace("file://", "")):
    raise FileNotFoundError(f"❌ 日志文件路径不存在: {logs_path}")

log_files = [os.path.join(logs_path.replace("file://", ""), f) for f in os.listdir(logs_path.replace("file://", "")) if f.endswith("-result.log")]

if not log_files:
    raise FileNotFoundError(f"❌ 未找到任何日志文件，请检查路径: {logs_path}")

log_df = spark.read.text(log_files)

# **10. 解析访问路径**
log_df = log_df.withColumn("VisitURL", regexp_extract(col("value"), r'"(https://library.xmu.edu.cn.*?)"', 1))

# **11. 统计每篇文章的访问量**
visit_count = log_df.groupBy("VisitURL").agg(count("*").alias("VisitCount"))

# **12. 结合文章数据，计算文章热度**
articles = articles.join(visit_count, articles["URL"] == visit_count["VisitURL"], "left").fillna(0)

# **13. 展示结果**
print("\n📌 Top 10 文章访问量：")
articles.select("Column", "Title", "Views", "VisitCount").orderBy(col("VisitCount").desc()).show(10)

print("\n📌 文章主题分析：")
topics.show(5, truncate=False)

# **14. 存储分析结果**
output_path = "file:///home/hadoop/Downloads/分析结果/"

articles.select("Column", "Title", "Views", "VisitCount").write.csv(output_path + "article_popularity.csv", header=True, mode="overwrite")
topics.write.csv(output_path + "article_topics.csv", header=True, mode="overwrite")

# **15. 关闭 Spark**
spark.stop()
