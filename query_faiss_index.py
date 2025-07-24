import numpy as np
import faiss
import json
import argparse
import re
import difflib
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher

def load_faiss_index(path='data/embeddings/faiss.index'):
    return faiss.read_index(path)

def load_metadata(path='data/embeddings/metadata.json'):
    with open(path, 'r') as f:
        return json.load(f)

def encode_query(query, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model.encode([query])[0]

def extract_function_name(function_source):
    """Extract function name from function source code."""
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

def normalize_text_for_matching(text):
    """Normalize text for better matching by handling camelCase, snake_case, etc."""
    # Convert to lowercase
    text_lower = text.lower()
    
    # Split by common delimiters and create variations
    variations = set()
    
    # Original text
    variations.add(text_lower)
    
    # Split by spaces
    variations.add(' '.join(text_lower.split()))
    
    # Split by underscores
    if '_' in text_lower:
        variations.add(' '.join(text_lower.split('_')))
    
    # Split camelCase (e.g., getClassifyData -> get classify data)
    import re
    camel_case_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    if camel_case_split != text:
        variations.add(camel_case_split.lower())
    
    # Split by hyphens
    if '-' in text_lower:
        variations.add(' '.join(text_lower.split('-')))
    
    # Remove all delimiters (e.g., getClassifyData -> getclassifydata)
    no_delimiters = re.sub(r'[_\-\s]', '', text_lower)
    variations.add(no_delimiters)
    
    return variations

def text_similarity_score(query, text):
    """Calculate enhanced text similarity with camelCase, snake_case, and space handling."""
    query_lower = query.lower().strip()
    text_lower = text.lower()
    
    # 1. Exact substring matching (highest priority)
    if query_lower in text_lower:
        return 1.0
    
    # 2. Normalize both query and text for better matching
    query_variations = normalize_text_for_matching(query)
    text_variations = normalize_text_for_matching(text)
    
    # 3. Check for exact matches in any variation
    for query_var in query_variations:
        for text_var in text_variations:
            if query_var == text_var:
                return 1.0
            if query_var in text_var or text_var in query_var:
                return 0.95
    
    # 4. Split query into words and clean
    query_words = [word.strip() for word in query_lower.split() if word.strip()]
    text_words = set(text_lower.split())
    
    # 5. Filter out common words and short words
    common_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
        'if', 'else', 'elif', 'try', 'except', 'finally', 'with', 'as', 'from', 'import',
        'def', 'class', 'return', 'pass', 'break', 'continue', 'raise', 'assert'
    }
    
    # Keep only meaningful words (not common, not too short)
    meaningful_words = [word for word in query_words if word not in common_words and len(word) > 2]
    
    # If no meaningful words, use all non-common words
    if not meaningful_words:
        meaningful_words = [word for word in query_words if word not in common_words]
    
    # If still no meaningful words, use all words
    if not meaningful_words:
        meaningful_words = query_words
    
    # 6. Enhanced word matching with normalization
    exact_matches = []
    partial_matches = []
    
    for query_word in meaningful_words:
        query_word_variations = normalize_text_for_matching(query_word)
        
        for text_word in text_words:
            text_word_variations = normalize_text_for_matching(text_word)
            
            # Check for exact matches in any variation
            for query_var in query_word_variations:
                for text_var in text_word_variations:
                    if query_var == text_var:
                        exact_matches.append(query_word)
                        break
                if query_word in exact_matches:
                    break
            
            # Check for partial matches
            if query_word not in exact_matches:
                for query_var in query_word_variations:
                    for text_var in text_word_variations:
                        if query_var in text_var or text_var in query_var:
                            partial_matches.append(query_word)
                            break
                    if query_word in partial_matches:
                        break
    
    # Remove duplicates
    exact_matches = list(set(exact_matches))
    partial_matches = list(set(partial_matches))
    
    # Calculate ratios
    exact_match_ratio = len(exact_matches) / len(meaningful_words) if meaningful_words else 0
    partial_match_ratio = len(partial_matches) / len(meaningful_words) if meaningful_words else 0
    
    # 7. Enhanced scoring with better handling of partial matches
    if exact_match_ratio == 1.0:
        return 1.0  # Perfect match
    elif exact_match_ratio >= 0.8:
        return 0.9  # Excellent match
    elif exact_match_ratio >= 0.6:
        return 0.7 + (0.1 * partial_match_ratio)  # Good match with partial boost
    elif exact_match_ratio >= 0.4:
        return 0.5 + (0.2 * partial_match_ratio)  # Moderate match with partial boost
    elif exact_match_ratio >= 0.2:
        return 0.3 + (0.2 * partial_match_ratio)  # Weak match with partial boost
    else:
        # Even with no exact matches, consider partial matches
        if partial_match_ratio >= 0.5:
            return 0.4  # Good partial match
        elif partial_match_ratio >= 0.3:
            return 0.2  # Moderate partial match
        else:
            return 0.0  # No meaningful match

def calculate_comprehensive_similarity(query, metadata, query_embedding, index):
    """Calculate both semantic and text similarity for EVERY item in the database."""
    print("Calculating semantic similarity for ALL items...")
    
    # Get semantic similarity for ALL items
    distances, indices = index.search(np.array([query_embedding]), len(metadata))
    semantic_scores = 1 / (1 + distances[0])
    
    print(f"Semantic calculation completed for {len(metadata)} items")
    
    print("Calculating text similarity for ALL items...")
    
    # Calculate text similarity for ALL items
    text_scores = []
    function_name_scores = []
    detailed_analysis = []
    
    # Extract meaningful words from query
    query_lower = query.lower()
    query_words = [word.strip() for word in query_lower.split() if word.strip()]
    common_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
        'if', 'else', 'elif', 'try', 'except', 'finally', 'with', 'as', 'from', 'import',
        'def', 'class', 'return', 'pass', 'break', 'continue', 'raise', 'assert'
    }
    meaningful_words = [word for word in query_words if word not in common_words and len(word) > 2]
    if not meaningful_words:
        meaningful_words = [word for word in query_words if word not in common_words]
    if not meaningful_words:
        meaningful_words = query_words
    
    for i, item in enumerate(metadata):
        # Extract function name
        function_name = extract_function_name(item['function_source'])
        
        # Calculate text similarity for combined content
        combined_text = f"{item['function_source']} {item['test_source']}"
        text_score = text_similarity_score(query, combined_text)
        
        # Calculate function name specific score with enhanced matching
        function_name_score = 0
        if function_name:
            function_name_score = text_similarity_score(query, function_name)
            
            # Additional boost for function name matches
            query_variations = normalize_text_for_matching(query)
            function_variations = normalize_text_for_matching(function_name)
            
            # Check for exact matches in any variation
            for query_var in query_variations:
                for func_var in function_variations:
                    if query_var == func_var:
                        function_name_score = max(function_name_score, 0.95)
                        break
        
        # Boost overall text score if function name matches well
        if function_name_score > text_score:
            text_score = max(text_score, function_name_score * 0.8)
        
        # Calculate sequence similarity for technical terms
        sequence_similarity = 0
        if meaningful_words:
            query_combined = ' '.join(meaningful_words)
            text_words = set(combined_text.lower().split())
            text_combined = ' '.join(list(text_words)[:50])  # Limit for performance
            sequence_similarity = SequenceMatcher(None, query_combined, text_combined).ratio()
        
        # Enhanced scoring with sequence similarity
        if text_score < 0.5 and sequence_similarity > 0.3:
            text_score = max(text_score, sequence_similarity * 0.6)
        
        text_scores.append(text_score)
        function_name_scores.append(function_name_score)
        
        # Store detailed analysis
        detailed_analysis.append({
            'index': i,
            'function_name': function_name,
            'text_score': text_score,
            'function_name_score': function_name_score,
            'semantic_score': semantic_scores[i],
            'sequence_similarity': sequence_similarity,
            'combined_text_length': len(combined_text)
        })
    
    print(f"Text calculation completed for {len(metadata)} items")
    
    return semantic_scores, text_scores, function_name_scores, detailed_analysis

def calculate_hybrid_scores(query, metadata, query_embedding, index, top_k=200):
    """Calculate comprehensive hybrid scores with multiple strategies and return raw similarities."""
    # Calculate comprehensive similarity for ALL items
    distances, indices = index.search(np.array([query_embedding]), top_k)
    # Cosine similarity: 1 / (1 + distance)
    raw_similarities = 1 / (1 + distances[0])
    
    semantic_scores, text_scores, function_name_scores, detailed_analysis = calculate_comprehensive_similarity(
        query, metadata, query_embedding, index
    )
    
    # Normalize scores
    max_semantic = np.max(semantic_scores) if len(semantic_scores) > 0 else 1
    max_text = max(text_scores) if text_scores else 1
    max_function = max(function_name_scores) if function_name_scores else 1
    
    if max_semantic > 0:
        semantic_scores = semantic_scores / max_semantic
    if max_text > 0:
        text_scores = [s / max_text for s in text_scores]
    if max_function > 0:
        function_name_scores = [s / max_function for s in function_name_scores]
    
    # Calculate multiple hybrid strategies
    comprehensive_scores = []
    
    for i in range(len(metadata)):
        # Strategy 1: Original strict prioritization
        if text_scores[i] >= 0.8:
            hybrid_score = 0.02 * semantic_scores[i] + 0.98 * text_scores[i]
        elif text_scores[i] >= 0.5:
            hybrid_score = 0.1 * semantic_scores[i] + 0.9 * text_scores[i]
        elif text_scores[i] >= 0.2:
            hybrid_score = 0.3 * semantic_scores[i] + 0.7 * text_scores[i]
        else:
            hybrid_score = 0.8 * semantic_scores[i] + 0.2 * text_scores[i]
            hybrid_score = hybrid_score * 0.5
        
        # Strategy 2: Balanced approach
        balanced_score = 0.5 * semantic_scores[i] + 0.5 * text_scores[i]
        
        # Strategy 3: Function name boosted (increased weight for function names)
        function_boosted = 0.2 * semantic_scores[i] + 0.3 * text_scores[i] + 0.5 * function_name_scores[i]
        
        comprehensive_scores.append({
            'index': i,
            'hybrid_score': hybrid_score,
            'balanced_score': balanced_score,
            'function_boosted': function_boosted,
            'semantic_score': semantic_scores[i],
            'text_score': text_scores[i],
            'function_name_score': function_name_scores[i],
            'raw_semantic_score': raw_similarities[i] if i < len(raw_similarities) else 0
        })
    
    # Sort by hybrid score (original strategy)
    comprehensive_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    # Get top results
    results = []
    for score_info in comprehensive_scores[:top_k]:
        idx = score_info['index']
        results.append({
            'metadata': metadata[idx],
            'semantic_score': float(score_info['semantic_score']),
            'text_score': score_info['text_score'],
            'hybrid_score': score_info['hybrid_score'],
            'balanced_score': score_info['balanced_score'],
            'function_boosted': score_info['function_boosted'],
            'function_name_score': score_info['function_name_score'],
            'raw_semantic_score': score_info['raw_semantic_score']
        })
    
    return results

def filter_by_relevance_threshold(results, min_hybrid_score=0.1, min_text_score=0.05):
    """Filter results by minimum relevance thresholds."""
    filtered_results = []
    
    for result in results:
        # Check if result meets minimum thresholds
        if (result['hybrid_score'] >= min_hybrid_score and 
            result['text_score'] >= min_text_score):
            filtered_results.append(result)
    
    return filtered_results

def query_faiss_index(index, query_embedding, metadata, query, top_k=5, min_hybrid_score=0.1, min_text_score=0.05, strategy='hybrid'):
    """Enhanced query function with comprehensive similarity analysis and multiple strategies."""
    results = calculate_hybrid_scores(query, metadata, query_embedding, index, top_k * 3)
    
    # Sort by selected strategy
    if strategy == 'balanced':
        results.sort(key=lambda x: x['balanced_score'], reverse=True)
    elif strategy == 'function_boosted':
        results.sort(key=lambda x: x['function_boosted'], reverse=True)
    else:  # hybrid (default)
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    # Filter by relevance threshold
    filtered_results = filter_by_relevance_threshold(results, min_hybrid_score, min_text_score)
    
    # If we have enough filtered results, return top_k
    if len(filtered_results) >= top_k:
        return filtered_results[:top_k]
    else:
        # If not enough filtered results, return what we have (up to top_k)
        return filtered_results[:top_k]

def debug_ranking(query, results, meaningful_words):
    """Debug function to show why results are ranked the way they are."""
    print(f"\n=== RANKING DEBUG ===")
    print(f"Query: '{query}'")
    print(f"Meaningful words: {meaningful_words}")
    print(f"Top {len(results)} results analysis:")
    
    for i, result in enumerate(results[:5]):  # Show top 5
        item = result['metadata']
        function_name = extract_function_name(item['function_source'])
        combined_text = f"{item['function_source']} {item['test_source']}"
        
        # Count meaningful word matches
        meaningful_matches = []
        for word in meaningful_words:
            if word in combined_text.lower():
                meaningful_matches.append(word)
        
        print(f"\n{i+1}. (Score: {result['hybrid_score']:.4f})")
        print(f"   Function: {function_name or 'Unknown'}")
        print(f"   Text Score: {result['text_score']:.4f}, Semantic Score: {result['semantic_score']:.4f}")
        print(f"   Meaningful words found: {meaningful_matches}")
        print(f"   Coverage: {len(meaningful_matches)}/{len(meaningful_words)} = {len(meaningful_matches)/len(meaningful_words)*100:.1f}%")
        print(f"   Source: {item['source_file']}")

def main():
    parser = argparse.ArgumentParser(description='Query the FAISS index for similar function-test pairs')
    parser.add_argument('--query', '-q', type=str, help='The query text to search for')
    parser.add_argument('--top_k', '-k', type=int, default=5, help='Number of results to return (default: 5)')
    parser.add_argument('--show_scores', '-s', action='store_true', help='Show detailed similarity scores')
    parser.add_argument('--debug', '-d', action='store_true', help='Show detailed ranking debug information')
    parser.add_argument('--min_hybrid_score', type=float, default=0.1, help='Minimum hybrid score threshold (default: 0.1)')
    parser.add_argument('--min_text_score', type=float, default=0.05, help='Minimum text score threshold (default: 0.05)')
    parser.add_argument('--strategy', type=str, default='function_boosted', 
                       choices=['hybrid', 'balanced', 'function_boosted'],
                       help='Ranking strategy (default: function_boosted)')
    args = parser.parse_args()

    if not args.query:
        args.query = input("Enter your query: ")

    print(f"\nQuerying for: {args.query}")
    print(f"Relevance thresholds: Hybrid >= {args.min_hybrid_score}, Text >= {args.min_text_score}")

    print("Loading FAISS index and metadata...")
    index = load_faiss_index()
    metadata = load_metadata()

    print("Encoding query...")
    query_embedding = encode_query(args.query)

    print(f"Querying with {args.strategy} strategy...")
    results = query_faiss_index(index, query_embedding, metadata, args.query, args.top_k, 
                               args.min_hybrid_score, args.min_text_score, args.strategy)

    # Extract meaningful words for debug
    query_lower = args.query.lower()
    query_words = [word.strip() for word in query_lower.split() if word.strip()]
    common_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
        'if', 'else', 'elif', 'try', 'except', 'finally', 'with', 'as', 'from', 'import',
        'def', 'class', 'return', 'pass', 'break', 'continue', 'raise', 'assert'
    }
    meaningful_words = [word for word in query_words if word not in common_words and len(word) > 2]
    if not meaningful_words:
        meaningful_words = [word for word in query_words if word not in common_words]
    if not meaningful_words:
        meaningful_words = query_words

    if args.debug:
        # Use the same calculation as the main function for consistency
        debug_results = calculate_hybrid_scores(args.query, metadata, query_embedding, index, args.top_k * 3)
        debug_ranking(args.query, debug_results, meaningful_words)

    print(f"\nTop {len(results)} results using {args.strategy} strategy (filtered by relevance thresholds):")
    if not results:
        print("No results met the relevance thresholds. Try lowering the thresholds or using a different query.")
        return
    
    # Check if we have any meaningful matches
    meaningful_match = False
    
    # Natural language support: keyword-based matching and function name boosting with fuzzy matching
    def get_keywords(text):
        stopwords = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
            'if', 'else', 'elif', 'try', 'except', 'finally', 'as', 'from', 'import',
            'def', 'class', 'return', 'pass', 'break', 'continue', 'raise', 'assert', 'function', 'method', 'handle', 'handles', 'handling'
        ])
        words = re.findall(r'\w+', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]

    def fuzzy_in(keyword, candidates, cutoff=0.7):
        # Returns True if keyword is close to any candidate string
        return bool(difflib.get_close_matches(keyword, candidates, n=1, cutoff=cutoff))

    query_keywords = get_keywords(args.query)
    prioritized_results = []
    # Increase candidate pool for keyword and function name prioritization
    for result in results[:200]:  # Check top 200 results for prioritization
        func_name = extract_function_name(result['metadata']['function_source'])
        combined_text = f"{result['metadata']['function_source']} {result['metadata']['test_source']}".lower()
        code_tokens = re.findall(r'\w+', combined_text)
        match_count = 0
        exact_fn_match = False
        partial_fn_match = False
        for kw in query_keywords:
            # Fuzzy match for function name
            if func_name and fuzzy_in(kw, [func_name.lower()], cutoff=0.95):
                exact_fn_match = True
            if func_name and fuzzy_in(kw, [func_name.lower()], cutoff=0.7):
                partial_fn_match = True
            # Fuzzy match for function name or code tokens
            if (func_name and fuzzy_in(kw, [func_name.lower()], cutoff=0.7)) or fuzzy_in(kw, code_tokens, cutoff=0.7):
                match_count += 1
        # Consider meaningful if at least half the keywords match
        if query_keywords and match_count >= max(1, len(query_keywords) // 2):
            meaningful_match = True
        prioritized_results.append((exact_fn_match, partial_fn_match, match_count, result))
    # Always show exact or fuzzy function name matches from the entire metadata
    for item in metadata:
        func_name = extract_function_name(item['function_source'])
        for kw in query_keywords:
            if func_name and fuzzy_in(kw, [func_name.lower()], cutoff=0.7):
                # Create a dummy result in the same format as others
                dummy_result = {
                    'metadata': item,
                    'semantic_score': 1.0,
                    'text_score': 1.0,
                    'hybrid_score': 1.0,
                    'balanced_score': 1.0,
                    'function_boosted': 1.0,
                    'function_name_score': 1.0
                }
                prioritized_results.insert(0, (True, True, len(query_keywords), dummy_result))
                meaningful_match = True
    if not meaningful_match:
        print("No relevant results found. Try rephrasing your query or check your data.")
        return
    # Sort: exact function name match first, then partial, then by match count
    prioritized_results.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    # Replace results with prioritized order
    results = [r[-1] for r in prioritized_results[:5]]

    for i, result in enumerate(results):
        print(f"\n{i+1}. (Hybrid Score: {result['hybrid_score']:.4f})")
        if args.show_scores:
            print(f"   Semantic Score: {result['semantic_score']:.4f}")
            print(f"   Text Score: {result['text_score']:.4f}")
            print(f"   Function Name Score: {result['function_name_score']:.4f}")
            print(f"   Balanced Score: {result['balanced_score']:.4f}")
            print(f"   Function Boosted Score: {result['function_boosted']:.4f}")
            print(f"   Raw Semantic Score: {result['raw_semantic_score']:.4f}")
        
        function_name = extract_function_name(result['metadata']['function_source'])
        if function_name:
            print(f"   Function Name: {function_name}")
        
        # Show meaningful word matches
        combined_text = f"{result['metadata']['function_source']} {result['metadata']['test_source']}"
        meaningful_matches = [word for word in meaningful_words if word in combined_text.lower()]
        if meaningful_matches:
            print(f"   Matches: {meaningful_matches}")
        
        print(f"   Function: {result['metadata']['function_source'][:200]}...")
        print(f"   Test: {result['metadata']['test_source'][:200]}...")
        print(f"   Source File: {result['metadata']['source_file']}")
        print("-" * 80)

if __name__ == '__main__':
    main() 