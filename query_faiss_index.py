import numpy as np
import faiss
import json
import argparse
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from thefuzz import fuzz

def load_faiss_index(path='data/embeddings/faiss.index'):
    """Loads the FAISS index from the specified path."""
    return faiss.read_index(path)

def load_metadata(path='data/embeddings/metadata.json'):
    """Loads the metadata from the specified path."""
    with open(path, 'r') as f:
        return json.load(f)

def encode_query(query, model_name='all-MiniLM-L6-v2'):
    """Encodes the query using the specified sentence-transformer model."""
    model = SentenceTransformer(model_name)
    return model.encode([query], convert_to_tensor=True)

def extract_function_name(function_source):
    """Extracts the function name from the function source code."""
    patterns = [
        r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]',
        r'async\s+def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    ]
    for pattern in patterns:
        match = re.search(pattern, function_source)
        if match:
            return match.group(1)
    return None

def calculate_text_similarity(query, text):
    """Calculates the text similarity using fuzzy matching."""
    return fuzz.partial_ratio(query, text) / 100.0

def hybrid_search(query, index, metadata, query_embedding, top_k=100):
    """Performs a hybrid search using semantic and text-based similarity."""
    # 1. Semantic Search (retrieval)
    distances, indices = index.search(query_embedding, top_k)
    
    retrieved_docs = [metadata[i] for i in indices[0]]
    
    # 2. Text-based Scoring (scoring)
    for i, doc in enumerate(retrieved_docs):
        combined_text = f"{doc['function_source']} {doc['test_source']}"
        text_score = calculate_text_similarity(query, combined_text)
        
        function_name = extract_function_name(doc['function_source'])
        function_name_score = 0
        if function_name:
            function_name_score = calculate_text_similarity(query, function_name)
        
        # Combine scores (you can experiment with different weighting)
        retrieved_docs[i]['semantic_score'] = 1 / (1 + distances[0][i])
        retrieved_docs[i]['text_score'] = text_score
        retrieved_docs[i]['function_name_score'] = function_name_score
        retrieved_docs[i]['hybrid_score'] = (0.4 * retrieved_docs[i]['semantic_score']) + \
                                            (0.4 * text_score) + \
                                            (0.2 * function_name_score)

    return retrieved_docs

def rerank_results(query, results, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', top_k=5):
    """Re-ranks the results using a cross-encoder model."""
    model = CrossEncoder(model_name)
    
    # Prepare pairs for the cross-encoder
    pairs = []
    for result in results:
        combined_text = f"Function: {result['function_source']}\nTest: {result['test_source']}"
        pairs.append([query, combined_text])
        
    # Get scores from the cross-encoder
    scores = model.predict(pairs)
    
    # Scale scores to a 0-1 range for better differentiation
    min_score, max_score = np.min(scores), np.max(scores)
    if max_score > min_score:
        scaled_scores = (scores - min_score) / (max_score - min_score)
    else:
        scaled_scores = np.zeros_like(scores)

    # Check if the query is likely a function name search (single word)
    is_name_query = ' ' not in query.strip()

    for i, result in enumerate(results):
        function_name = extract_function_name(result['function_source'])
        rerank_score = scaled_scores[i]
        
        function_name_sim = 0
        if function_name:
            function_name_sim = fuzz.ratio(query.lower(), function_name.lower()) / 100.0
        
        if is_name_query:
            # For single-word queries, the score is mostly the name similarity.
            # The rerank_score from the semantic search acts as a small boost.
            final_score = function_name_sim * 0.9 + rerank_score * 0.1
        else:
            # For multi-word queries, blend the scores
            adjustment = (function_name_sim - 0.5) * 0.5
            final_score = rerank_score + adjustment

        # If the function name is an exact match, push the score to the top
        if function_name and query.strip().lower() == function_name.strip().lower():
            final_score = 1.0
            
        results[i]['rerank_score'] = np.clip(final_score, 0, 1)
        
    # Sort by the new final score
    results.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    # Filter out results with very low scores, especially for name queries
    if is_name_query:
        results = [r for r in results if r['rerank_score'] > 0.3]

    return results[:top_k]

def main():
    parser = argparse.ArgumentParser(description='Query the FAISS index for similar function-test pairs')
    parser.add_argument('--query', '-q', type=str, help='The query text to search for')
    parser.add_argument('--top_k', '-k', type=int, default=5, help='Number of results to return (default: 5)')
    parser.add_argument('--show_scores', '-s', action='store_true', help='Show detailed similarity scores')
    args = parser.parse_args()

    if not args.query:
        args.query = input("Enter your query: ")

    print(f"\nQuerying for: {args.query}")

    print("Loading FAISS index and metadata...")
    index = load_faiss_index()
    metadata = load_metadata()

    print("Encoding query...")
    query_embedding = encode_query(args.query)

    print("Performing hybrid search...")
    hybrid_results = hybrid_search(args.query, index, metadata, query_embedding.cpu().numpy(), top_k=200)

    print("Re-ranking results...")
    final_results = rerank_results(args.query, hybrid_results, top_k=args.top_k)

    print(f"\nTop {args.top_k} results:")
    
    displayed_results = 0
    for result in final_results:
        if displayed_results >= args.top_k:
            break
            
        function_name = extract_function_name(result['function_source'])
        source_file = result['source_file']
        
        displayed_results += 1
        
        print(f"\n{displayed_results}. (Rerank Score: {result['rerank_score']:.4f})")
        if args.show_scores:
            print(f"   Hybrid Score: {result['hybrid_score']:.4f}")
            print(f"   Semantic Score: {result['semantic_score']:.4f}")
            print(f"   Text Score: {result['text_score']:.4f}")
            print(f"   Function Name Score: {result['function_name_score']:.4f}")
        
        print(f"   Function Name: {function_name}")
        print(f"   Function: {result['function_source'][:200]}...")
        print(f"   Test: {result['test_source'][:200]}...")
        print(f"   Source File: {result['source_file']}")
        print("-" * 80)

if __name__ == '__main__':
    main()