import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant

# Use the updated HuggingFaceEmbeddings class
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url = "http://localhost:6333"

# Instantiate QdrantClient correctly
client = QdrantClient(url=url)

print(client)
print("##############")

# Use the Qdrant class from langchain_qdrant
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

print(db)
print("######")

query = "What is Metastatic disease?"

docs = db.similarity_search_with_score(query=query, k=2)
for i in docs:
    doc, score = i
    # Use print with utf-8 encoding
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})
