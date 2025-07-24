import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# This script generates embeddings for function-test pairs extracted from Python files.
# It uses the SentenceTransformer library to create embeddings and saves them to a file.
def load_pairs():
    """Load function-test pairs from JSON file."""
    with open('data/pairs/function_test_pairs.json', 'r') as f:
        pairs = json.load(f)
        # Print the structure of the first pair
        if pairs:
            print("Structure of first pair:", json.dumps(pairs[0], indent=2))
        return pairs

# This function generates embeddings for function-test pairs using the SentenceTransformer library.
# It combines the function source code and test source code into a single text for each pair,
# and then encodes these texts into embeddings.
def generate_embeddings(pairs, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings for function-test pairs using sentence-transformers."""
    # Initialize the model
    model = SentenceTransformer(model_name)
    
    # Prepare texts for embedding
    texts = []
    for pair in pairs:
        # Combine function and test into a single text
        text = f"Function: {pair['function_source']}\nTest: {pair['test_source']}"
        texts.append(text)
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings

# This function saves the generated embeddings and metadata (function-test pairs) to files.
# It creates a directory for embeddings if it doesn't exist, saves the embeddings as a NumPy array,
# and saves the metadata as a JSON file.
def save_embeddings(embeddings, pairs):
    """Save embeddings and metadata to files."""
    # Create directory if it doesn't exist
    os.makedirs('data/embeddings', exist_ok=True)
    
    # Save embeddings
    np.save('data/embeddings/embeddings.npy', embeddings)
    
    # Save metadata (function-test pairs)
    with open('data/embeddings/metadata.json', 'w') as f:
        json.dump(pairs, f, indent=2)

def main():
    # Load pairs
    print("Loading function-test pairs...")
    pairs = load_pairs()
    
    # Generate embeddings
    embeddings = generate_embeddings(pairs)
    
    # Save results
    print("Saving embeddings and metadata...")
    save_embeddings(embeddings, pairs)
    
    print(f"Generated embeddings for {len(pairs)} pairs.")
    print("Results saved to data/embeddings/")

if __name__ == '__main__':
    main() 