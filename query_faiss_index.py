import numpy as np
import faiss
import json
import argparse
import re  # needed by normalize_query and regex heuristics
from sentence_transformers import SentenceTransformer, CrossEncoder
from thefuzz import fuzz
from tree_sitter_extractor import TreeSitterExtractor

def load_faiss_index(path='data/embeddings/faiss.index'):
    """Loads the FAISS index from the specified path."""
    return faiss.read_index(path)

def load_metadata(path='data/embeddings/metadata.json'):
    """Loads the metadata from the specified path."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def encode_query(query, model_name='all-MiniLM-L6-v2'):
    """Encodes the query using the specified sentence-transformer model."""
    model = SentenceTransformer(model_name)
    return model.encode([query], convert_to_tensor=True)

def extract_function_name(function_source):
    """Extracts the method name from the C# method source code using regex (fast heuristic)."""
    try:
        pattern = (
            r'(?:public|private|protected|internal)\s+'
            r'(?:static\s+)?(?:virtual\s+)?(?:override\s+)?(?:async\s+)?'
            r'(?:void|Task|bool|int|string|float|double|Vector2|Vector3|Vector4|Quaternion|'
            r'GameObject|Transform|MonoBehaviour|ScriptableObject|T|object)\s+'
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        )
        match = re.search(pattern, (function_source or ""))
        if match:
            return match.group(1)
    except Exception as e:
        print(f"Function name extraction failed: {e}")
    return None

def normalize_query(query):
    """
    Normalizes a query by replacing likely variable names with a generic 'VAR' token.
    This helps treat queries like "a + b" and "x + y" as semantically identical.
    """
    PLACEHOLDER_VARS = {'a', 'b', 'c', 'i', 'j', 'k', 'n', 'x', 'y', 'z'}
    parts = re.split(r'([^\w])', query)
    normalized_parts = []
    for part in parts:
        if part.isalnum() and part.lower() in PLACEHOLDER_VARS:
            normalized_parts.append('VAR')
        else:
            normalized_parts.append(part)
    return "".join(normalized_parts)

def calculate_text_similarity(query, text):
    """Calculates the text similarity using fuzzy matching."""
    return fuzz.partial_ratio(query or "", text or "") / 100.0

# ----------------- Filtering helpers -----------------
_TEST_ATTR_RE = re.compile(
    r'\[(?:Test|TestCase|TestCaseSource|UnityTest|Fact|Theory|TestMethod|DataTestMethod)\b.*?\]',
    re.IGNORECASE | re.DOTALL
)

def is_test_file(path: str) -> bool:
    """Heuristic: whether a source file looks like a test file."""
    if not path:
        return False
    p = path.lower()
    return any(tok in p for tok in [
        '/tests/', '\\tests\\', '/test/', '\\test\\', '.tests.', '.test.',
        'test_', '_test', 'tests.cs', 'test.cs'
    ]) or p.endswith('tests.cs') or p.endswith('test.cs')

def is_test_method(function_source: str, source_file: str) -> bool:
    """
    Heuristic: whether a method is itself a unit test method.
    Checks attributes, filename, and common naming.
    """
    src = function_source or ""
    if _TEST_ATTR_RE.search(src):
        return True
    if is_test_file(source_file or ""):
        return True
    name = extract_function_name(src) or ""
    if name.startswith("Test") or name.startswith("Tests") or name.startswith("Should"):
        return True
    return False

def has_linked_test(doc: dict) -> bool:
    """Whether this doc has a non-empty associated test snippet."""
    ts = doc.get('test_source')
    return isinstance(ts, str) and ts.strip() != ""
# -----------------------------------------------------

def score_and_annotate_docs(query, docs, distances_row):
    """Compute hybrid/text/name scores for already-retrieved docs."""
    for i, doc in enumerate(docs):
        combined_text = f"{doc.get('function_source','')} {doc.get('test_source','')}"
        text_score = calculate_text_similarity(query, combined_text)

        function_name = extract_function_name(doc.get('function_source', ''))
        function_name_score = 0.0
        if function_name:
            function_name_score = calculate_text_similarity(query, function_name)

        sem = 1.0 / (1.0 + float(distances_row[i]))
        doc['semantic_score'] = sem
        doc['text_score'] = text_score
        doc['function_name_score'] = function_name_score
        doc['hybrid_score'] = 0.4 * sem + 0.4 * text_score + 0.2 * function_name_score
    return docs

def retrieve(index, metadata, query_embedding, k):
    """Helper to run FAISS search and pull docs."""
    distances, indices = index.search(query_embedding, k)
    docs = [metadata[i] for i in indices[0]]
    return distances[0], docs

def filtered_hybrid_search(query, index, metadata, query_embedding, top_k=200):
    """
    Retrieve top_k, score them, then filter:
    - keep only production methods (not tests)
    - that have a linked test snippet
    """
    distances_row, retrieved_docs = retrieve(index, metadata, query_embedding, top_k)
    retrieved_docs = score_and_annotate_docs(query, retrieved_docs, distances_row)
    filtered = [
        d for d in retrieved_docs
        if has_linked_test(d) and not is_test_method(d.get('function_source', ''), d.get('source_file', ''))
    ]
    return filtered

def fallback_similar_with_tests(query, index, metadata, query_embedding,
                                expand_ks=(500, 1000, 2000)):
    """
    If initial strict retrieval returns nothing, expand the retrieval pool
    and try again to find *similar* methods that do have linked tests.
    Stops at the first non-empty result set.
    """
    for k in expand_ks:
        print(f"No matches after initial filter; expanding search to top_k={k} ...")
        distances_row, retrieved_docs = retrieve(index, metadata, query_embedding, k)
        retrieved_docs = score_and_annotate_docs(query, retrieved_docs, distances_row)
        candidates = [
            d for d in retrieved_docs
            if has_linked_test(d) and not is_test_method(d.get('function_source', ''), d.get('source_file', ''))
        ]
        if candidates:
            print(f"Found {len(candidates)} similar methods with tests at k={k}.")
            return candidates
    return []

def rerank_results(query, results, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
                   top_k=5, keep_name_results=False):
    """Re-ranks the results using a cross-encoder model and applies safe name-query filtering."""
    if not results:
        return []

    model = CrossEncoder(model_name)

    is_name_query = ' ' not in (query or '').strip()

    # For descriptive queries, normalize to help the model focus on structure
    if not is_name_query:
        query_for_encoder = normalize_query(query or "")
    else:
        query_for_encoder = query or ""

    # Prepare pairs for the cross-encoder
    pairs = []
    for result in results:
        combined_text = f"Method: {result.get('function_source','')}\nTest: {result.get('test_source','')}"
        pairs.append([query_for_encoder, combined_text])

    # Cross-encoder scores
    scores = model.predict(pairs)

    # Scale to 0-1
    min_score, max_score = float(np.min(scores)), float(np.max(scores))
    if max_score > min_score:
        scaled_scores = (scores - min_score) / (max_score - min_score)
    else:
        scaled_scores = np.zeros_like(scores, dtype=float)

    # Code-like query detection
    is_code_like_query = any(op in (query or '') for op in ['+', '-', '*', '/', '==', '!=', '<', '>'])

    for i, result in enumerate(results):
        function_name = extract_function_name(result.get('function_source', ''))
        rerank_score = float(scaled_scores[i])

        function_name_sim = 0.0
        if function_name:
            function_name_sim = fuzz.ratio((query or '').lower(), function_name.lower()) / 100.0

        if is_code_like_query:
            code_sim = fuzz.token_set_ratio(query or "", result.get('function_source', '')) / 100.0
            final_score = code_sim * 0.8 + rerank_score * 0.1 + function_name_sim * 0.1
        elif is_name_query:
            final_score = function_name_sim * 0.9 + rerank_score * 0.1
        else:
            adjustment = (function_name_sim - 0.5) * 0.5
            final_score = rerank_score + adjustment

        if function_name and (query or '').strip().lower() == function_name.strip().lower():
            final_score = 1.0

        result['rerank_score'] = float(np.clip(final_score, 0.0, 1.0))

    # Sort by the new final score
    results.sort(key=lambda x: x['rerank_score'], reverse=True)

    # Safer name-query filtering (never drop everything)
    if is_name_query and not keep_name_results:
        name_min = 0.01  # lowered from 0.10
        pre = len(results)
        filtered = [r for r in results if r['rerank_score'] >= name_min]
        if not filtered:
            filtered = results[:top_k]  # fallback to top_k if all are below threshold
            print(f"Name-query filter would have removed all results; "
                  f"returning top {len(filtered)} by score instead.")
        else:
            print(f"Filtering name query results. Found {pre} before filtering. "
                  f"After filtering: {len(filtered)} results remain (threshold={name_min}).")
        results = filtered

    return results[:top_k]

def main():
    parser = argparse.ArgumentParser(description='Query the FAISS index for similar function-test pairs')
    parser.add_argument('--query', '-q', type=str, help='The query text to search for')
    parser.add_argument('--top_k', '-k', type=int, default=5, help='Number of results to return (default: 5)')
    parser.add_argument('--show_scores', '-s', action='store_true', help='Show detailed similarity scores')
    parser.add_argument('--keep_name_results', action='store_true',
                        help='Do not filter low-scoring name queries')
    parser.add_argument('--no_fallback', action='store_true',
                        help='Disable expanding search when nothing matches the strict filter')
    args = parser.parse_args()

    if not args.query:
        args.query = input("Enter your query: ")

    query = args.query
    print(f"\nQuerying for: {query}")

    # Normalize for semantic search if descriptive
    if ' ' in (query or '').strip():
        query_for_semantic_search = normalize_query(query)
        if query_for_semantic_search != query:
            print(f"Normalized query for semantic search: {query_for_semantic_search}")
    else:
        query_for_semantic_search = query

    print("Loading FAISS index and metadata...")
    index = load_faiss_index()
    metadata = load_metadata()

    print("Encoding query...")
    query_embedding = encode_query(query_for_semantic_search)

    print("Performing hybrid search...")
    # First pass: strict retrieval with filter
    hybrid_results = filtered_hybrid_search(
        query, index, metadata, query_embedding.cpu().numpy(), top_k=200
    )

    # Fallback: expand retrieval to find similar methods with tests
    if not hybrid_results and not args.no_fallback:
        hybrid_results = fallback_similar_with_tests(
            query, index, metadata, query_embedding.cpu().numpy(),
            expand_ks=(500, 1000, 2000)
        )

    if not hybrid_results:
        print("\nNo matching non-test methods with linked tests were found.")
        print("Tip: Try a shorter query (e.g., just the method name) or remove very specific constants.")
        return

    print("Re-ranking results...")
    final_results = rerank_results(
        query, hybrid_results, top_k=args.top_k,
        keep_name_results=getattr(args, 'keep_name_results', False)
    )

    if not final_results:
        print("\nNo results after reranking.")
        return

    print(f"\nTop {args.top_k} results (non-test methods that have tests):")
    displayed_results = 0
    for result in final_results:
        if displayed_results >= args.top_k:
            break

        function_name = extract_function_name(result.get('function_source', ''))
        source_file = result.get('source_file', '')

        displayed_results += 1

        print(f"\n{displayed_results}. (Rerank Score: {result['rerank_score']:.4f})")
        if args.show_scores:
            print(f"   Hybrid Score: {result.get('hybrid_score', 0.0):.4f}")
            print(f"   Semantic Score: {result.get('semantic_score', 0.0):.4f}")
            print(f"   Text Score: {result.get('text_score', 0.0):.4f}")
            print(f"   Function Name Score: {result.get('function_name_score', 0.0):.4f}")

        print(f"   Method Name: {function_name}")
        print(f"   Method: {result.get('function_source','')[:200]}...")
        print(f"   Test (linked): {result.get('test_source','')[:200]}...")
        print(f"   Source File: {source_file}")
        print("-" * 80)

if __name__ == '__main__':
    main()
