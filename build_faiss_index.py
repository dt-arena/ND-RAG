import numpy as np
import faiss
import os
import json


# This script builds a FAISS index from precomputed embeddings.
# It loads the embeddings and metadata from files, creates a FAISS index,
# and saves the index to a file for efficient similarity search.
def load_embeddings():
    embeddings = np.load('data/embeddings/embeddings.npy')
    with open('data/embeddings/metadata.json', 'r') as f:
        metadata = json.load(f)
    return embeddings, metadata

# This function builds a FAISS index from the embeddings.
# It uses the L2 distance metric for similarity search. 
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# This function saves the FAISS index to a specified path.
# The default path is 'data/embeddings/faiss.index'.    
def save_faiss_index(index, path='data/embeddings/faiss.index'):
    faiss.write_index(index, path)


def main():
    print('Loading embeddings and metadata...')
    embeddings, metadata = load_embeddings()
    print(f'Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.')

    print('Building FAISS index...')
    index = build_faiss_index(embeddings)

    print('Saving FAISS index...')
    os.makedirs('data/embeddings', exist_ok=True)
    save_faiss_index(index)
    print('FAISS index saved to data/embeddings/faiss.index')

if __name__ == '__main__':
    main() 