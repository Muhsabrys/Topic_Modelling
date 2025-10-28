!pip install bertopic sentence-transformers pandas

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd

#loading your text
file_path = "mytext.txt"

with open(file_path, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if len(line.strip()) > 10]

print(f"✅ Loaded {len(texts)} cleaned posts")

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

topic_model = BERTopic(
    embedding_model=embedding_model,
    language="english",
    min_topic_size=5,
    verbose=True
)

topics, probs = topic_model.fit_transform(texts)

topic_info = topic_model.get_topic_info()
print("\n🧩 Top Topics:")
print(topic_info.head(10))

topic_info.to_csv("alzheimers_topics_summary.csv", index=False)
print("💾 Topic summary saved to: alzheimers_topics_summary.csv")


doc_info = topic_model.get_document_info(texts)
doc_info.to_csv("alzheimers_topic_assignments.csv", index=False)
print("💾 Document-topic mapping saved to: alzheimers_topic_assignments.csv")

#Generating Interactive Visualizations

topic_model.visualize_topics().write_html("viz_topics_map.html")
topic_model.visualize_barchart(top_n_topics=15).write_html("viz_topics_barchart.html")
topic_model.visualize_hierarchy().write_html("viz_topics_hierarchy.html")
topic_model.visualize_heatmap().write_html("viz_topics_heatmap.html")

print("done!")
