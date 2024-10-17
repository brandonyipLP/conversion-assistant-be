import os
from google.cloud import aiplatform
from google.oauth2 import service_account
import pandas as pd
import numpy as np
from typing import List, Dict
import textwrap

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the JSON file in the scripts directory
json_path = os.path.join(script_dir, 'genial-caster-438919-i8-71baa2f7add1.json')

# Set up Google Cloud credentials
credentials = service_account.Credentials.from_service_account_file(json_path)
aiplatform.init(credentials=credentials, project='genial-caster-438919-i8')

# Function to chunk text
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split the input text into overlapping chunks.
    """
    chunks = textwrap.wrap(text, chunk_size, break_long_words=False)
    
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk = chunks[i-1][-overlap:] + chunk
        if i < len(chunks) - 1:
            chunk = chunk + chunks[i+1][:overlap]
        overlapped_chunks.append(chunk)
    
    return overlapped_chunks

# Function to generate embeddings using text-embedding-004 model
def generate_embedding(text: str) -> List[float]:
    """
    Generate embeddings using Google's text-embedding-004 model.
    """
    model = aiplatform.TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    embeddings = model.get_embeddings([text])
    return embeddings[0].values

# Function to create and populate the vector database
def setup_vector_database(input_file: str, user_id: str):
    """
    Create and populate the Vertex AI Vector Database with embeddings from the input file.
    """
    # Create a new index if it doesn't exist
    try:
        index = aiplatform.MatchingEngineIndex.list(filter=f"display_name=customer_data_index")[0]
        print("Using existing index.")
    except IndexError:
        print("Creating new index.")
        index = aiplatform.MatchingEngineIndex.create(
            display_name="customer_data_index",
            dimensions=768,  # Dimension of text-embedding-004 model
            approximate_neighbors_count=50,
            distance_measure_type="COSINE_DISTANCE",
            description="Index for customer-specific data"
        )

    # Read the input file
    with open(input_file, 'r') as file:
        text = file.read()

    # Chunk the text
    chunks = chunk_text(text)

    # Generate embeddings and create index points
    index_points = []
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        index_points.append(
            aiplatform.IndexDatapoint(
                datapoint_id=f"{user_id}_{i}",
                feature_vector=embedding,
                restricts={'user_id': user_id}
            )
        )

    # Insert the index points into the database
    index.upsert_datapoints(index_points)

    print(f"Vector database populated with {len(index_points)} datapoints for user {user_id}.")

# Function to query the vector database
def query_vector_database(query: str, user_id: str, top_k: int = 5) -> List[Dict]:
    """
    Query the vector database with the given query and return top_k results.
    """
    index = aiplatform.MatchingEngineIndex.list(filter=f"display_name=customer_data_index")[0]
    query_embedding = generate_embedding(query)

    matched_items = index.find_neighbors(
        query_vector=query_embedding,
        num_neighbors=top_k,
        restricts={'user_id': user_id}
    )

    results = []
    for item in matched_items:
        results.append({
            'id': item.id,
            'distance': item.distance,
            'restricts': item.restricts
        })

    return results

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the input_file.txt in the scripts directory
input_file_path = os.path.join(script_dir, 'input_file.txt')

# Update the input_file variable
input_file = input_file_path

if __name__ == "__main__":
    # Example usage
    user_id = "example_user_123"
    
    # Set up the vector database with the input file
    setup_vector_database(input_file, user_id)
    
    # Example query
    query = "What is the main topic of the document?"
    results = query_vector_database(query, user_id)
    
    print("Quereey results:")
    for result in results:
        print(f"ID: {result['id']}, Distance: {result['distance']}")
