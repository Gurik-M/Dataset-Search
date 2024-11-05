# IMPORTING DATA INTO PINECONE

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset
import time
from tqdm.auto import tqdm

def init_pinecone(api_key):
    return Pinecone(api_key=api_key)

def create_index(pc, index_name, dimension, region="us-east-1"):
    """Create a new Pinecone index if it doesn't exist, return original if it does exist"""
    spec = ServerlessSpec(cloud="aws", region=region)
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name,
            dimension=dimension,
            metric='dotproduct',
            spec=spec
        )
        # wait for index to be initialized
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    
    return pc.Index(index_name)

def batch_upsert(index, texts, model, batch_size=32):
    """Upsert texts to Pinecone index in batches"""
    for i in tqdm(range(0, len(texts), batch_size)):
        # set end position of batch
        i_end = min(i+batch_size, len(texts))
        # get batch of lines and IDs
        lines_batch = texts[i: i_end]
        ids_batch = [str(n) for n in range(i, i_end)]
        # create embeddings
        embeds = model.encode(lines_batch)
        # prep metadata and upsert batch
        meta = [{'text': line} for line in lines_batch]
        to_upsert = zip(ids_batch, embeds, meta)
        # upsert to Pinecone
        index.upsert(vectors=list(to_upsert))
        print(f"Done! Upserted batch {i//batch_size} of {len(texts)//batch_size}")

def query_index(index, query_text, model, top_k=5):
    """Query the index with a text string"""
    xq = model.encode(query_text).tolist()
    results = index.query(
        vector=xq,
        top_k=top_k,
        include_values=True,
        include_metadata=True
    )
    return results

def index_trec_dataset(api_key, index_name, model_name='all-MiniLM-L6-v2', sample_size=1000):
    """Index TREC dataset into Pinecone"""
    # Initialize model and Pinecone
    model = SentenceTransformer(model_name)
    pc = init_pinecone(api_key)
    
    # Get or create index
    index = create_index(pc, index_name, dimension=384)
    
    # Load dataset
    trec = load_dataset('trec', split=f'train[:{sample_size}]')
    
    # Index the data
    batch_upsert(index, trec['text'], model)

    print("Trec: ", trec[0])
    
    return index

def index_custom_dataset(api_key, index_name, texts, model_name='all-MiniLM-L6-v2'):
    """Index custom list of texts into Pinecone"""
    # Initialize model and Pinecone
    model = SentenceTransformer(model_name)
    pc = init_pinecone(api_key)
    
    # Get or create index
    index = create_index(pc, index_name, dimension=384)
    
    # Index the data
    batch_upsert(index, texts, model)
    
    return index