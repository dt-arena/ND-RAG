# query_semantic.py
#!/usr/bin/env python3
"""
Semantic Query System for Function-Test Pairs
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import re
import numpy as np
import faiss
from typing import List, Dict
from tqdm import tqdm
import difflib

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed, using fallback")

class SemanticQuerySystem:
    def _init_(self, embeddings_path: str = 'data/embeddings'):
        """Initialize the semantic query system"""
        self.embeddings_path = embeddings_path
        self.index = None
        self.metadata = None
        self.concept_index = None
        self.name_index = None
        self.model = None
        
        # Load everything
        self.load_index()
        self.load_metadata()
        self.load_model()
        
        # Domain knowledge (same as in generate_embeddings.py)
        self.init_domain_knowledge()
    
    def init_domain_knowledge(self):
        """Initialize domain knowledge for concept mapping"""
        self.concept_groups = {
            'collision': ['collider', 'collision', 'physics', 'rigidbody', 'trigger', 'overlap', 'hit', 'contact'],
            'click': ['click', 'clicked', 'tap', 'press', 'select', 'mouse', 'pointer', 'input', 'button'],
            'handler': ['handler', 'handle', 'callback', 'listener', 'event', 'on', 'process', 'respond'],
            'transform': ['transform', 'position', 'rotation', 'scale', 'translate', 'rotate', 'move'],
            'render': ['render', 'draw', 'display', 'graphics', 'mesh', 'material', 'shader'],
            'update': ['update', 'tick', 'frame', 'loop', 'fixed', 'late'],
            'animation': ['animation', 'animate', 'anim', 'clip', 'animator', 'blend'],
            'shape': ['circle', 'square', 'rectangle', 'triangle', 'box', 'sphere', 'capsule', 'mesh'],
            'ui': ['button', 'text', 'image', 'panel', 'canvas', 'slider'],
            'audio': ['sound', 'music', 'audio', 'listener', 'source'],
            'input': ['input', 'key', 'mouse', 'touch', 'joystick']
        }
        
        # Build reverse mapping
        self.term_to_concepts = {}
        for concept, terms in self.concept_groups.items():
            for term in terms:
                if term not in self.term_to_concepts:
                    self.term_to_concepts[term] = []
                self.term_to_concepts[term].append(concept)
    
    def load_index(self):
        """Load FAISS index"""
        index_path = os.path.join(self.embeddings_path, 'faiss.index')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        # Store index dimension to pick a compatible embedding model later
        self.index_dim = getattr(self.index, 'd', None)
        print(f"Loaded index with {self.index.ntotal} vectors")
    
    def load_metadata(self):
        """Load metadata including concept and name indices"""
        metadata_path = os.path.join(self.embeddings_path, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        # Support both dict-based metadata (with indices) and raw list of pairs
        self.metadata = data.get('pairs', []) if isinstance(data, dict) else data
        
        # Build indices if not present
        if 'concept_index' not in data or 'name_index' not in data:
            print("Building concept and name indices...")
            self.concept_index = {}
            self.name_index = {}
            
            for idx, item in tqdm(enumerate(self.metadata), total=len(self.metadata)):
                func_name = (item.get('function_name') or '').strip()
                if func_name:
                    key = func_name.lower()
                    if key not in self.name_index:
                        self.name_index[key] = []
                    self.name_index[key].append(idx)
                    
                    # Index decomposed name parts for concept-style lookups
                    for part in self.decompose_name(func_name):
                        if part not in self.concept_index:
                            self.concept_index[part] = []
                        self.concept_index[part].append(idx)
        else:
            self.concept_index = data.get('concept_index', {})
            self.name_index = data.get('name_index', {})
        
        print(f"Loaded {len(self.metadata)} function-test pairs")
        print(f"Concept index: {len(self.concept_index)} concepts")
        print(f"Name index: {len(self.name_index)} function names")
    
    def load_model(self):
        """Load the embedding model"""
        if not HAS_SENTENCE_TRANSFORMERS:
            print("Sentence transformers not available, semantic search limited")
            return
        
        try:
            # Choose model based on FAISS index dimension if available
            # Common dims: 384 (all-MiniLM-L6-v2), 768 (mpnet-base)
            preferred_model = None
            if hasattr(self, 'index_dim') and self.index_dim is not None:
                if self.index_dim == 384:
                    preferred_model = 'all-MiniLM-L6-v2'
                elif self.index_dim == 768:
                    preferred_model = 'all-mpnet-base-v2'
            
            if preferred_model:
                self.model = SentenceTransformer(preferred_model)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                print(f"✓ Model loaded to match index dim {self.index_dim}: {preferred_model}")
            else:
                # Fallback to UniXCoder, then MiniLM
                self.model = SentenceTransformer('microsoft/unixcoder-base')
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                print("✓ UniXCoder model loaded")
        except:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = 384
                print("✓ MiniLM model loaded (fallback)")
            except Exception as e:
                print(f"Could not load embedding model: {e}")
                self.model = None
    
    def decompose_name(self, name: str) -> List[str]:
        """Decompose camelCase/PascalCase names"""
        if not name:
            return []
        
        # Split on capitals, underscores, numbers
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]+$|\d+', name)
        return [p.lower() for p in parts if p]
    
    def create_query_embedding(self, query: str) -> np.ndarray:
        """Create embedding for query"""
        if not self.model:
            # Fallback: create random embedding
            return np.random.randn(768)
        
        # Decompose if it looks like a function name
        parts = self.decompose_name(query)
        
        # Build query description
        description_parts = []
        
        if parts:
            description_parts.append(f"Function name: {query}")
            description_parts.append(f"Name components: {' '.join(parts)}")
            
            # Add related concepts
            concepts = set()
            for part in parts:
                if part in self.term_to_concepts:
                    concepts.update(self.term_to_concepts[part])
                concepts.add(part)
            
            if concepts:
                description_parts.append(f"Key concepts: {' '.join(concepts)}")
        else:
            # Treat as code or description
            description_parts.append(f"Code: {query}")
        
        full_description = " | ".join(description_parts)
        
        # Generate embedding
        embedding = self.model.encode(full_description)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def search_by_name(self, query: str, top_k: int = 20) -> List[Dict]:
        """Search by function name matching"""
        query_lower = query.lower()
        results = []
        
        # Exact name matches
        if query_lower in self.name_index:
            for idx in self.name_index[query_lower]:
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['match_score'] = 1.0
                    result['match_type'] = 'exact_name'
                    result['index'] = idx
                    results.append(result)
        
        # Decompose query name
        query_parts = self.decompose_name(query)
        
        # Search by concepts
        for part in query_parts:
            if part in self.concept_index:
                for idx in self.concept_index[part]:
                    if idx < len(self.metadata):
                        # Check if already in results
                        if not any(r['index'] == idx for r in results):
                            result = self.metadata[idx].copy()
                            result['match_score'] = 0.7
                            result['match_type'] = 'concept'
                            result['index'] = idx
                            results.append(result)
        
        # Expanded concepts
        for part in query_parts:
            related_concepts = self.term_to_concepts.get(part, [])
            related_terms = set()
            for concept in related_concepts:
                related_terms.update(self.concept_groups.get(concept, []))
            related_terms.add(part)  # include original
            
            for term in related_terms:
                if term in self.concept_index:
                    for idx in self.concept_index[term]:
                        if not any(r['index'] == idx for r in results):
                            result = self.metadata[idx].copy()
                            result['match_score'] = 0.65  # slightly lower than direct concept
                            result['match_type'] = 'expanded_concept'
                            result['index'] = idx
                            results.append(result)
        
        # Partial name matches (substring)
        for name, indices in self.name_index.items():
            if query_lower != name:
                # Check substring match
                if query_lower in name or name in query_lower:
                    for idx in indices:
                        if idx < len(self.metadata):
                            if not any(r['index'] == idx for r in results):
                                result = self.metadata[idx].copy()
                                result['match_score'] = 0.5
                                result['match_type'] = 'partial_name'
                                result['index'] = idx
                                results.append(result)
        
        # Fuzzy name matches
        close_names = difflib.get_close_matches(query_lower, list(self.name_index.keys()), n=50, cutoff=0.5)
        for close_name in close_names:
            similarity = difflib.SequenceMatcher(None, query_lower, close_name).ratio()
            for idx in self.name_index[close_name]:
                if not any(r['index'] == idx for r in results):
                    result = self.metadata[idx].copy()
                    result['match_score'] = similarity * 0.8
                    result['match_type'] = 'fuzzy_name'
                    result['index'] = idx
                    results.append(result)
        
        # Part fuzzy matches
        for part in query_parts:
            for name, indices in self.name_index.items():
                name_parts = self.decompose_name(name)
                for n_part in name_parts:
                    similarity = difflib.SequenceMatcher(None, part, n_part).ratio()
                    if similarity > 0.55:
                        for idx in indices:
                            if not any(r['index'] == idx for r in results):
                                result = self.metadata[idx].copy()
                                result['match_score'] = similarity * 0.6
                                result['match_type'] = 'part_fuzzy'
                                result['index'] = idx
                                results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x['match_score'], reverse=True)
        
        return results[:top_k]
    
    def search_semantic(self, query: str, top_k: int = 20) -> List[Dict]:
        """Semantic search using embeddings"""
        if not self.model:
            print("Warning: No embedding model loaded, falling back to name search")
            return self.search_by_name(query, top_k)
        
        # Create query embedding
        query_embedding = self.create_query_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['match_score'] = float(dist)
                result['match_type'] = 'semantic'
                result['index'] = idx
                results.append(result)
        
        return results
    
    def query(self, query: str, top_k: int = 5, only_with_tests: bool = True) -> List[Dict]:
        """Main query method combining multiple strategies"""
        
        # Detect query type
        is_name_like = bool(re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', query))
        
        all_results = []
        
        if is_name_like:
            print(f"Query appears to be a function name: {query}")
            # Prioritize name-based search
            name_results = self.search_by_name(query, top_k * 5)
            all_results.extend(name_results)
        
        # Always add semantic search
        semantic_results = self.search_semantic(query, top_k * 5)
        
        # Merge results (avoid duplicates)
        seen_indices = set(r['index'] for r in all_results)
        for result in semantic_results:
            if result['index'] not in seen_indices:
                all_results.append(result)
                seen_indices.add(result['index'])
        
        before_filter = len(all_results)
        # Filter by test availability
        if only_with_tests:
            all_results = [r for r in all_results if r.get('test_source') and r['test_source'].strip()]
        
        after_filter = len(all_results)
        if only_with_tests and after_filter < before_filter:
            print(f"Filtered from {before_filter} to {after_filter} (only with tests)")
        
        # Sort by score and limit
        all_results.sort(key=lambda x: x['match_score'], reverse=True)
        
        return all_results[:top_k]
    
    def format_result(self, result: Dict, index: int, verbose: bool = False) -> str:
        """Format a result for display"""
        lines = []
        
        lines.append(f"\n{'='*60}")
        lines.append(f"Result #{index + 1} - {result.get('match_type', '').upper()} MATCH")
        lines.append(f"Score: {result.get('match_score', 0):.4f}")
        lines.append(f"{'='*60}")
        
        # Function details
        lines.append(f"\nFunction: {result.get('function_name', 'Unknown')}")
        
        # Show semantic features if available
        if verbose and 'semantic_features' in result:
            features = result['semantic_features']
            if features.get('concepts'):
                lines.append(f"Concepts: {', '.join(features['concepts'][:10])}")
            if features.get('patterns'):
                lines.append(f"Patterns: {', '.join(features['patterns'])}")
        
        # Function source
        lines.append("\n--- Function Code ---")
        func_source = result.get('function_source', 'No source available')
        if len(func_source) > 500 and not verbose:
            func_source = func_source[:500] + "\n... (truncated)"
        lines.append(func_source)
        
        # Test source
        if result.get('test_source'):
            lines.append("\n--- Test Code ---")
            test_source = result['test_source']
            if len(test_source) > 400 and not verbose:
                test_source = test_source[:400] + "\n... (truncated)"
            lines.append(test_source)
        else:
            lines.append("\n--- No Test Available ---")
        
        return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description='Semantic code search for function-test pairs')
    parser.add_argument('--query', '-q', type=str, help='Query string (function name or code)')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results')
    parser.add_argument('--all', action='store_true', help='Include functions without tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--embeddings-path', default='data/embeddings', help='Path to embeddings')
    
    args = parser.parse_args()
    
    # Get query
    if not args.query:
        args.query = input("Enter query: ").strip()
    
    if not args.query:
        print("No query provided")
        return
    
    print(f"\n{'='*60}")
    print("SEMANTIC CODE SEARCH")
    print(f"{'='*60}")
    print(f"Query: {args.query}")
    
    # Initialize system
    try:
        system = SemanticQuerySystem(args.embeddings_path)
    except Exception as e:
        print(f"Error initializing system: {e}")
        print("Make sure you've run generate_embeddings.py first")
        return
    
    # Search
    results = system.query(
        query=args.query,
        top_k=args.top_k,
        only_with_tests=not args.all
    )
    
    # Display results
    if results:
        print(f"\n✓ Found {len(results)} results")
        for i, result in enumerate(results):
            print(system.format_result(result, i, args.verbose))
    else:
        print("\n✗ No results found")
        if not args.all:
            print("Tip: Try --all to include functions without tests")

if __name__ == '__main__':
    main()