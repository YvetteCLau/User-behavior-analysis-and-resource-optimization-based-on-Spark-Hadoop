from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, collect_set
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

# **✅ 1. 创建 SparkSession**
spark = SparkSession.builder \
    .appName("Academic Collaboration Network") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# **✅ 2. 读取数据**
data_path = "file:///home/hadoop/Downloads/图书馆数据/学者库数据/"
output_path = "file:///home/hadoop/Downloads/科研合作数据/"

# **自动读取所有 CSV 文件**
article_data = None
csv_files = ["ArticleCn.csv", "ArticleEn.csv"]
for file in csv_files:
    file_path = data_path + file
    temp_df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    if article_data is None:
        article_data = temp_df
    else:
        article_data = article_data.union(temp_df)

print(f"\n✅ 读取 {len(csv_files)} 个 CSV 文件，总数据量: {article_data.count()} 行")

# ✅ **随机抽样 2% 的数据**
article_data = article_data.sample(fraction=0.02, seed=42)
print(f"\n✅ 抽样后数据量: {article_data.count()} 行")

# ✅ **拆分作者**
article_data = article_data.withColumn("Author", explode(split(col("作者"), "[;,]")))
article_data = article_data.dropna(subset=["Author", "论文分类"])

# ✅ **统计合作关系**
coauthor_pairs = article_data.groupBy("论文题名").agg(collect_set("Author").alias("Authors"))

# ✅ **构建学者合作网络**
authors = article_data.select("Author").distinct().toPandas()
author_index = {name: idx for idx, name in enumerate(authors["Author"])}
edges = []

for row in coauthor_pairs.collect():
    author_list = list(set(row["Authors"]))
    for i in range(len(author_list)):
        for j in range(i + 1, len(author_list)):
            edges.append((author_index.get(author_list[i]), author_index.get(author_list[j])))

# ✅ **仅保留合作次数 ≥5 的学者**
collaboration_counts = {}
for a, b in edges:
    collaboration_counts[a] = collaboration_counts.get(a, 0) + 1
    collaboration_counts[b] = collaboration_counts.get(b, 0) + 1

filtered_edges = [(a, b) for a, b in edges if collaboration_counts[a] >= 5 and collaboration_counts[b] >= 5]
print("\n📌 过滤后边的数量:", len(filtered_edges))

edge_index = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()

# ✅ **创建 `Embedding`**
embedding_dim = 8  # 降低维度，减少计算量
x = torch.nn.Embedding(len(author_index), embedding_dim)

# **✅ 3. 构建 PyTorch Geometric 数据**
data = Data(x=x.weight, edge_index=edge_index)

# **✅ 4. 定义 GCN 预测模型**
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ✅ **训练 GCN**
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\n📌 使用设备:", device)

data = data.to(device)
model = GCN(in_channels=embedding_dim, hidden_channels=8, out_channels=embedding_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

batch_size = 128  # 进一步降低 batch_size，减少计算压力
for epoch in range(30):  # 训练 30 轮（减少训练时间）
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.mse_loss(out, data.x)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# ✅ **获取学者嵌入**
model.eval()
embeddings = model(data.x, data.edge_index).cpu().detach().numpy()

# **✅ 5. 计算学术领域相似度**
categories = article_data.select("Author", "论文分类").distinct().toPandas()
category_dict = categories.groupby("Author")["论文分类"].apply(lambda x: " ".join(x)).to_dict()

# ✅ **仅计算前 2000 位学者**
author_list = list(category_dict.keys())[:2000]
vectorizer = TfidfVectorizer()
category_matrix = vectorizer.fit_transform([category_dict[a] for a in author_list])
similarity_matrix = cosine_similarity(category_matrix)

# ✅ **保存合作推荐结果**
recommendations_df = pd.DataFrame(similarity_matrix, index=author_list, columns=author_list)
recommendations_df.to_csv(os.path.join(output_path.replace("file://", ""), "科研合作推荐.csv"))

# **✅ 6. 设置 Matplotlib 中文字体**
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文乱码
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# **✅ 7. 可视化**
plt.figure(figsize=(12, 5))

# ✅ **合作网络**
G = nx.Graph()
for a, b in filtered_edges:
    G.add_edge(authors["Author"].iloc[a], authors["Author"].iloc[b])

plt.subplot(1, 2, 1)
nx.draw(G, with_labels=False, node_size=10, edge_color="gray")
plt.title("学者合作网络（抽样）")

# ✅ **学术领域相似度热力图**
plt.subplot(1, 2, 2)
sns.heatmap(similarity_matrix[:50, :50], cmap="Blues")
plt.title("学术领域相似度热力图（前 50 名学者）")

plt.show()

# ✅ **保存合作网络数据**
graph_df = pd.DataFrame(filtered_edges, columns=["Author_1", "Author_2"])
graph_df.to_csv(os.path.join(output_path.replace("file://", ""), "合作网络.csv"), index=False)

# ✅ **关闭 Spark**
spark.stop()
