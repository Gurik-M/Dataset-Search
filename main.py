from sentence_transformers import SentenceTransformer
import vectordb

# Configuration
API_KEY = "YOUR_API_KEY"
INDEX_NAME = 'semantic-search-mini-lm'
MODEL_NAME = 'all-MiniLM-L6-v2'

# Initialize model and Pinecone
model = SentenceTransformer(MODEL_NAME)
pc = vectordb.init_pinecone(API_KEY)

# Get or create index
index = vectordb.create_index(pc, INDEX_NAME, dimension=384)

# Example query
query = "what is the difference between a vector database and a semantic search engine?"
results = vectordb.query_index(index, query, model)

# Print results
for match in results['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")