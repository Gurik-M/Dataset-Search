# Semantic Dataset Search with Pinecone and Sentence Transformers
This project showcases a semantic search pipeline using Pinecone as the vector database and Sentence Transformers (in particular the all-MiniLM-L6-v2 model) for generating embeddings. With this setup, you can index large textual datasets, query them semantically, and retrieve the most relevant rows or content based on similarity to your query.

## Overview
Semantic search goes beyond traditional keyword-based lookups. By converting text into embeddings (vector representations), we can measure similarity between query embeddings and indexed text embeddings. This approach allows you to find results that conceptually match your query, even if they do not share the same exact keywords.

This project uses:
 - Sentence Transformers for generating embeddings.
 - Pinecone for storing and querying those embeddings.

## Features and Capabilities
 - Semantic Indexing: Convert text into embeddings and store them in Pinecone, enabling sub-second similarity search.
 - Scalable Vector Database: Utilize Pinecone’s cloud-based vector database to handle large datasets efficiently.
 - Flexible Deployment: Use the scripts locally or in a cloud environment with minimal changes.
 - Custom Dataset Support: Easily extend indexing to any text-based datasets (e.g., TREC).

## Project Files
#### main.py
1. Configuration: Contains Pinecone API Key, Index Name, and Model Name.
2. Initialization: Loads the Sentence Transformer model and initializes Pinecone.
3. Index Retrieval: Attempts to retrieve or create a Pinecone index.
4. Example Query: Demonstrates how to query the index and print out top matches.

#### vectordb.py
Contains helper functions and workflows:
1. ```init_pinecone()```: Initializes the Pinecone client.
2. ```create_index()```: Creates a new Pinecone index if it doesn’t already exist; otherwise, retrieves the existing one.
3. ```batch_upsert()```: Performs batch insert/upsert operations of text embeddings into the index.
4. ```query_index()```: Queries the index with a given text string and returns the closest matching documents.
5. ```index_trec_dataset()```: Convenience function to index a sample of the TREC dataset.
6. ```index_custom_dataset()```: Similar function but for custom data.

## Installation and Setup
1. Clone this repo
2. Install the required python packages
3. Create a Pinecone account and obtain an API Key if you haven’t already. You will need this in main.py (replace the placeholder value).

## Usage
There are 2 steps in this project, the first is to index the data, and the second is to query that index.

#### 1. Indexing Data
In vectordb.py, there is a helper function ```index_trec_dataset()``` which will fetch the TREC dataset, encode each text entry, and upsert the embeddings into a Pinecone index. TREC is just a example placeholder dataset, this can be modified to be any custom text dataset.

#### 2. Querying the Index
In main.py, you’ll find an example that performs the query "what is the difference between a vector database and a semantic search engine?". It works by first using the function ```vectordb.query_index()``` to take in your query string, encodes it with the Sentence Transformer model, and retrieve the top relevant texts from Pinecone. Then ```results['matches']``` is a list of matches with scores and metadata (such as the original text).

## Customization
1. Model Choice: By default, all-MiniLM-L6-v2 is used. You can change the model by updating the MODEL_NAME in main.py or passing a different model to the indexing functions in vectordb.py.
2. Indexing Additional Fields: If you have more metadata than just text, you can update the batch_upsert() function to store additional fields.
3. Similarity Metric: The default metric is dotproduct. You can switch to cosine or euclidean by updating the create_index() call in vectordb.py.
4. Batch Size: Tweak the batch_size argument in batch_upsert() to optimize performance for your environment.

## Deployment
There are 3 deployment options. In my opinion, a container deployment with Docker and AWS ECS would be the simplest and most straightforward.
1. Local/Dev Environment:
 - Run main.py after configuring your Pinecone API key.
 - Make sure your environment has all dependencies installed.
2. Cloud Deployment (e.g., on AWS, Azure, GCP, or other platforms):
 - Containerize the application (e.g., using Docker) if needed.
 - Store Pinecone credentials securely using environment variables or a secrets manager.
 - Make sure inbound/outbound networking is allowed for connecting to Pinecone’s API endpoints.
3. Serverless:
 - These scripts can be built into a serverless function with AWS Lambda as long as you install the necessary
   Python dependencies into your function environment (this can be done by zipping all dependencies into a single folder).