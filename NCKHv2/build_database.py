import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load your data
print("Loading data...")
df = pd.read_csv('public_train.csv')

# 2. Create the knowledge base from real news (label == 0)
# We also drop any rows where the post_message is empty.
knowledge_base_df = df[df['label'] == 0].dropna(subset=['post_message'])

# 3. Get the list of articles (we'll call them 'documents')
documents = knowledge_base_df['post_message'].tolist()

print(f"Created a knowledge base with {len(documents)} articles.")

# 4. Load an embedding model
# This model is great because it understands multiple languages, including Vietnamese.
print("Loading embedding model...")
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 5. Create the embeddings (numerical representations of the documents)
print("Creating embeddings for the knowledge base...")
doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True, show_progress_bar=True)

# 6. Build the FAISS index for fast searching
print("Building FAISS index...")
# We need to move embeddings to the CPU and convert to a numpy array for FAISS
doc_embeddings_np = doc_embeddings.cpu().numpy()

# The dimension of our vectors is the second value in the shape tuple
d = doc_embeddings_np.shape[1] 
index = faiss.IndexFlatL2(d)
index.add(doc_embeddings_np) # type: ignore

print("FAISS index built successfully!")

# --- Save the index and documents for later ---
# This is so you don't have to re-run the embedding process every time.
faiss.write_index(index, "knowledge_base.index")
with open("documents.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc + "\n")

print("Knowledge base saved to 'knowledge_base.index' and 'documents.txt'")