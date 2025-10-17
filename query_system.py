#!/usr/bin/env python3
"""
    Query System for Function-Test Pairs
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import re
import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import difflib

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed, using fallback")


class SemanticQuerySystem:
    def __init__(self, embeddings_path: str = 'data/embeddings', infer_tests: bool = False, infer_threshold: float = 1.2):
        self.embeddings_path = embeddings_path
        self.index = None
        self.index_dim = None
        self.metadata = None
        self.concept_index = None
        self.name_index = None
        self.model = None

        # inference controls
        self.infer_tests = infer_tests  # default OFF; enable with --allow-inferred-tests
        self.infer_threshold = infer_threshold

        # lookups built from metadata
        self._test_items_by_repo: Dict[str, list] = {}

        self.load_index()
        self.load_metadata()
        self.load_model()
        self.init_domain_knowledge()

    def init_domain_knowledge(self):
        self.concept_groups = {
            'collision': ['collider', 'collision', 'physics', 'rigidbody', 'trigger', 'overlap', 'hit', 'contact', 'ignore', 'ignorecollision'],
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
        self.term_to_concepts = {}
        for concept, terms in self.concept_groups.items():
            for term in terms:
                self.term_to_concepts.setdefault(term, []).append(concept)

    # ---- repo name extraction ----
    def _extract_repo_name(self, item: Dict) -> str:
        for key in ['repo_name', 'repository_name', 'repo', 'repository', 'project', 'source_repo']:
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                name = val.strip().rstrip('/').rsplit('/', 1)[-1]
                if name.endswith('.git'):
                    name = name[:-4]
                return name
        for key in ['repo_url', 'repository_url', 'git_url', 'origin_url']:
            url = (item.get(key) or '')
            if isinstance(url, str) and url:
                m = re.search(r'(?:^|/)(?P<repo>[^/]+?)(?:\.git)?/?$', url)
                if m:
                    return m.group('repo')
        for key in ['function_path', 'file_path', 'path', 'source_path']:
            p = (item.get(key) or '')
            if isinstance(p, str) and p:
                parts = [seg for seg in re.split(r'[\\/]+', p) if seg]
                if parts:
                    for seg in parts[:3]:
                        if seg.lower() not in {'src', 'source', 'assets', 'code', 'lib', 'include', 'unity', 'project'}:
                            return seg
                    return parts[0]
        return 'unknown'
    # ---------------------------------------------------------------------------------------------------------------------

    def load_index(self):
        index_path = os.path.join(self.embeddings_path, 'faiss.index')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.index = faiss.read_index(index_path)
        self.index_dim = getattr(self.index, 'd', None)

    def _build_test_lookup(self):
        """Build a repo->list of 'test-like' items for later inference."""
        self._test_items_by_repo = {}
        for it in self.metadata:
            repo = (it.get('repo_name') or self._extract_repo_name(it) or 'unknown').lower()
            if self._is_test_like_function(it):
                self._test_items_by_repo.setdefault(repo, []).append(it)

    def load_metadata(self):
        metadata_path = os.path.join(self.embeddings_path, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.metadata = data.get('pairs', []) if isinstance(data, dict) else data

        # ensure repo_name present
        for i, item in enumerate(self.metadata):
            if not isinstance(item.get('repo_name'), str) or not item.get('repo_name').strip():
                self.metadata[i]['repo_name'] = self._extract_repo_name(item)

        # indices
        if isinstance(data, dict) and ('concept_index' in data and 'name_index' in data):
            self.concept_index = data.get('concept_index', {})
            self.name_index = data.get('name_index', {})
        else:
            self.concept_index, self.name_index = {}, {}
            for idx, item in tqdm(enumerate(self.metadata), total=len(self.metadata)):
                func_name = (item.get('function_name') or '').strip()
                if func_name:
                    key = func_name.lower()
                    self.name_index.setdefault(key, []).append(idx)
                    for part in self.decompose_name(func_name):
                        self.concept_index.setdefault(part, []).append(idx)

        # build fast lookup for test inference
        self._build_test_lookup()

    def load_model(self):
        if not HAS_SENTENCE_TRANSFORMERS:
            print("Sentence transformers not available, semantic search limited")
            return
        try:
            preferred_model = None
            if getattr(self, 'index_dim', None) == 384:
                preferred_model = 'all-MiniLM-L6-v2'
            elif getattr(self, 'index_dim', None) == 768:
                preferred_model = 'all-mpnet-base-v2'
            if preferred_model:
                self.model = SentenceTransformer(preferred_model)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            else:
                self.model = SentenceTransformer('microsoft/unixcoder-base')
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = 384
            except Exception as e:
                print(f"Could not load embedding model: {e}")
                self.model = None

    def decompose_name(self, name: str) -> List[str]:
        if not name:
            return []
        words = re.split(r'\s+', name.strip())
        parts = []
        for word in words:
            parts.extend(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]+$|\d+', word))
        return [p.lower() for p in parts if p]

    def is_code_like(self, text: str) -> bool:
        code_indicators = [';', '{', '}', '(', ')', '=', 'return', 'void', 'public', 'int', 'string', 'if', 'for']
        return any(ind in text for ind in code_indicators) or len(text.split('\n')) > 1

    def is_name_like(self, text: str) -> bool:
        words = re.split(r'\s+', text.strip())
        return all(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', w) for w in words)

    def clean_code(self, code: str) -> str:
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        code = re.sub(r'\s+', ' ', code).strip()
        return code

    def calculate_name_similarity(self, query: str, func_name: str) -> float:
        return difflib.SequenceMatcher(None, query.lower(), func_name.lower()).ratio()

    def calculate_token_overlap(self, query: str, code: str) -> float:
        query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        code_tokens = set(re.findall(r'\b\w+\b', code.lower()))
        inter = len(query_tokens & code_tokens)
        union = len(query_tokens | code_tokens)
        return inter / union if union > 0 else 0.0

    def create_query_embedding(self, query: str) -> np.ndarray:
        if not self.model:
            return np.random.randn(768)
        if self.is_code_like(query):
            full_description = f"Code: {self.clean_code(query)}"
        else:
            parts = self.decompose_name(query)
            description_parts = []
            if parts:
                description_parts.append(f"Function name: {query}")
                description_parts.append(f"Name components: {' '.join(parts)}")
                concepts = set()
                for part in parts:
                    if part in self.term_to_concepts:
                        concepts.update(self.term_to_concepts[part])
                    concepts.add(part)
                if concepts:
                    description_parts.append(f"Key concepts: {' '.join(concepts)}")
            else:
                description_parts.append(f"Code: {query}")
            full_description = " | ".join(description_parts)
        emb = self.model.encode(full_description)
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        return emb

    def search_by_name(self, query: str, top_k: int = 20) -> List[Dict]:
        ql = query.lower()
        results = []
        if ql in self.name_index:
            for idx in self.name_index[ql]:
                if idx < len(self.metadata):
                    r = self.metadata[idx].copy()
                    r['match_score'] = 1.0
                    r['match_type'] = 'exact_name'
                    r['index'] = idx
                    results.append(r)
        parts = self.decompose_name(query)
        for part in parts:
            if part in self.concept_index:
                for idx in self.concept_index[part]:
                    if idx < len(self.metadata) and not any(r['index'] == idx for r in results):
                        r = self.metadata[idx].copy()
                        r['match_score'] = 0.7
                        r['match_type'] = 'concept'
                        r['index'] = idx
                        results.append(r)
        for part in parts:
            related_concepts = self.term_to_concepts.get(part, [])
            related_terms = set()
            for c in related_concepts:
                related_terms.update(self.concept_groups.get(c, []))
            related_terms.add(part)
            for term in related_terms:
                if term in self.concept_index:
                    for idx in self.concept_index[term]:
                        if not any(r['index'] == idx for r in results):
                            r = self.metadata[idx].copy()
                            r['match_score'] = 0.65
                            r['match_type'] = 'expanded_concept'
                            r['index'] = idx
                            results.append(r)
        for name, indices in self.name_index.items():
            if ql != name and (ql in name or name in ql):
                for idx in indices:
                    if idx < len(self.metadata) and not any(r['index'] == idx for r in results):
                        r = self.metadata[idx].copy()
                        r['match_score'] = 0.5
                        r['match_type'] = 'partial_name'
                        r['index'] = idx
                        results.append(r)
        close_names = difflib.get_close_matches(ql, list(self.name_index.keys()), n=50, cutoff=0.3)
        for close_name in close_names:
            sim = difflib.SequenceMatcher(None, ql, close_name).ratio()
            for idx in self.name_index[close_name]:
                if not any(r['index'] == idx for r in results):
                    r = self.metadata[idx].copy()
                    r['match_score'] = sim * 0.8
                    r['match_type'] = 'fuzzy_name'
                    r['index'] = idx
                    results.append(r)
        for part in parts:
            for name, indices in self.name_index.items():
                for n_part in self.decompose_name(name):
                    sim = difflib.SequenceMatcher(None, part, n_part).ratio()
                    if sim > 0.5:
                        for idx in indices:
                            if not any(r['index'] == idx for r in results):
                                r = self.metadata[idx].copy()
                                r['match_score'] = sim * 0.6
                                r['match_type'] = 'part_fuzzy'
                                r['index'] = idx
                                results.append(r)
        results.sort(key=lambda x: x['match_score'], reverse=True)
        return results

    def search_semantic(self, query: str, top_k: int = 20) -> List[Dict]:
        if not self.model:
            print("Warning: No embedding model loaded, falling back to name search")
            return self.search_by_name(query, top_k)
        q = self.create_query_embedding(query).reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(q, min(top_k, self.index.ntotal))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                r = self.metadata[idx].copy()
                r['match_score'] = float(dist)
                r['match_type'] = 'semantic'
                r['index'] = idx
                results.append(r)
        return results

    def re_rank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        for r in results:
            name_sim = self.calculate_name_similarity(query, r.get('function_name', ''))
            token_over = self.calculate_token_overlap(query, r.get('function_source', ''))
            semantic_score = r['match_score'] if r['match_type'] == 'semantic' else 0.5
            combined = 0.4 * semantic_score + 0.3 * name_sim + 0.3 * token_over
            r['hybrid_score'] = combined
            r['score_breakdown'] = {'semantic': semantic_score, 'name_sim': name_sim, 'token_over': token_over}
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return results

    # --- helper: detect test-like functions ---
    def _is_test_like_function(self, item: Dict) -> bool:
        fname = (item.get('function_name') or '').lower()
        fsrc = (item.get('function_source') or '').lower()
        fpath = (item.get('function_path') or '').lower()
        repo = (item.get('repo_name') or '').lower()

        path_has_test = any(s in fpath for s in ['/test', '\\test', '/tests', '\\tests'])
        name_has_test = (
            fname.startswith('test') or
            fname.endswith('test') or
            ' test' in fname or
            re.search(r'\btest\w*', fname) is not None
        )
        code_has_test_attr_or_assert = ('[test]' in fsrc) or ('assert.' in fsrc) or ('assert(' in fsrc)
        repo_indicates_tests = 'tests' in repo or repo.endswith('-tests') or repo.endswith('.tests')

        return path_has_test or name_has_test or code_has_test_attr_or_assert or repo_indicates_tests
    # ------------------------------------------

    # --- helper: scoring utilities for inference ---
    def _token_set(self, s: str) -> set:
        return set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', s.lower()))

    def _token_overlap(self, a: str, b: str) -> float:
        ta, tb = self._token_set(a), self._token_set(b)
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        return inter / union if union > 0 else 0.0
    # ---------------------------------------------

    # --- helper: infer a test for a function result if missing ---
    def _infer_test_for_result(self, r: Dict) -> Optional[Tuple[str, float, str]]:
        """
        Return (test_source, confidence, provenance='inferred') or None.
        Confidence blends:
          +1.0 if function name exact boundary match in test text
          +0.5 if function name case-insensitive substring
          + token overlap up to +0.5
          +0.2 if typical test markers exist in candidate
          +0.3 if file basename overlap
        """
        repo = (r.get('repo_name') or '').lower()
        func_name = (r.get('function_name') or '')
        func_name_lower = func_name.lower()
        if not repo or not self.infer_tests:
            return None
        candidates = self._test_items_by_repo.get(repo, [])
        if not candidates:
            return None

        ref_src = (r.get('function_source') or '')

        fpath = (r.get('function_path') or '')
        file_base = ''
        if fpath:
            parts = re.split(r'[\\/]', fpath)
            if parts:
                file_base = parts[-1].split('.')[0].lower()

        best = None
        best_score = 0.0
        for it in candidates:
            c_text = (it.get('test_source') or it.get('function_source') or '')
            if not c_text.strip():
                continue

            score = 0.0
            # name signals
            if func_name and re.search(r'\b' + re.escape(func_name) + r'\b', c_text):
                score += 1.0
            elif func_name_lower and func_name_lower in c_text.lower():
                score += 0.5

            # token overlap (bounded)
            overlap = self._token_overlap(ref_src, c_text)
            score += min(0.5, overlap)

            # path / filename hint
            tpath = ((it.get('function_path') or '') + ' ' + (it.get('file_path') or '')).lower()
            if file_base and file_base in tpath:
                score += 0.3

            # test markers
            if '[test]' in c_text.lower() or 'assert.' in c_text or 'assert(' in c_text:
                score += 0.2

            # guardrail: require some overlap unless we have a strong name match
            if overlap < 0.08 and score < 1.0:
                continue

            if score > best_score:
                best_score = score
                best = c_text

        if best and best_score >= self.infer_threshold:
            return best, best_score, 'inferred'
        return None
    # -------------------------------------------------------------

    def _attach_inferred_tests(self, items: List[Dict]) -> None:
        for r in items:
            if not r.get('test_source') or not r['test_source'].strip():
                inferred = self._infer_test_for_result(r)
                if inferred:
                    test_text, conf, prov = inferred
                    r['test_source'] = test_text
                    r['test_provenance'] = prov
                    r['test_confidence'] = conf
                else:
                    r['test_provenance'] = 'none'
                    r['test_confidence'] = 0.0
            else:
                r['test_provenance'] = 'direct'
                r['test_confidence'] = 1.0

    def query(self, query: str, top_k: int = 5, only_with_tests: bool = True, require_tests_strict: bool = False) -> List[Dict]:
        is_name_like = self.is_name_like(query)
        is_code_like_query = self.is_code_like(query)

        # Primary pool
        all_results: List[Dict] = []
        if is_name_like or not is_code_like_query:
            all_results.extend(self.search_by_name(query, top_k * 20))
        all_results_sem = self.search_semantic(query, top_k * 40)  # larger semantic pool
        seen = set(r['index'] for r in all_results)
        for r in all_results_sem:
            if r['index'] not in seen:
                all_results.append(r)
                seen.add(r['index'])

        # Rank
        all_results = self.re_rank_results(query, all_results)
        # Remove test functions from the main pool
        all_results = [r for r in all_results if not self._is_test_like_function(r)]

        # Stage A: nearest items that ALREADY have tests in metadata (direct provenance).
        direct_candidates = [r for r in all_results if r.get('test_source') and r['test_source'].strip()]
        if direct_candidates:
            for r in direct_candidates:
                r['test_provenance'] = 'direct'
                r['test_confidence'] = 1.0
            return direct_candidates[:top_k]

        # Stage B: broaden the semantic pool aggressively and again prefer items that already have tests
        expanded_sem_k = min(self.index.ntotal, 5000)  # widen pool to improve chance of finding test-bearing neighbors
        expanded_sem = self.search_semantic(query, top_k=expanded_sem_k)
        seen2 = set(r['index'] for r in all_results)
        for r in expanded_sem:
            if r['index'] not in seen2:
                all_results.append(r)
                seen2.add(r['index'])
        all_results = self.re_rank_results(query, all_results)
        all_results = [r for r in all_results if not self._is_test_like_function(r)]
        direct_candidates = [r for r in all_results if r.get('test_source') and r['test_source'].strip()]
        if direct_candidates:
            for r in direct_candidates:
                r['test_provenance'] = 'direct'
                r['test_confidence'] = 1.0
            return direct_candidates[:top_k]

        # Stage C (optional): allow inferred tests only if explicitly enabled
        if self.infer_tests:
            self._attach_inferred_tests(all_results)
            inferred = [r for r in all_results if r.get('test_source') and r['test_provenance'] == 'inferred']
            if inferred:
                return inferred[:top_k]

        # If strict tests required but none found, return empty rather than misleading items without tests
        if only_with_tests:
            return []

        # Last resort: return top_k nearest (without tests)
        return all_results[:top_k]

    def format_result(self, result: Dict, index: int, verbose: bool = False, show_tests: bool = True) -> str:
        lines = []
        lines.append(f"Result {index + 1}.")
        lines.append(f"Repo_name: {result.get('repo_name', 'unknown')}")
        lines.append(f"Function: {result.get('function_name', 'Unknown')}")

        func_source = result.get('function_source', '')
        if not func_source:
            lines.append("Function Code: (missing)")
        else:
            if not verbose and len(func_source) > 1000:
                func_source = func_source[:1000] + "\n... (truncated)"
            lines.append("Function Code:")
            lines.append(func_source)

        if show_tests:
            prov = result.get('test_provenance', 'none')
            conf = result.get('test_confidence', 0.0)
            note = ""
            if prov == 'direct' and result.get('match_type') not in ('exact_name',):
                note = " (similar function w/ direct test)"

            if prov == 'inferred':
                lines.append(f"Testcase: (inferred, conf={conf:.2f})")
            elif prov == 'direct':
                lines.append(f"Testcase: (direct){note}")
            else:
                lines.append("Testcase:")

            test_source = result.get('test_source', '')
            if not test_source:
                lines.append("No test source available")
            else:
                if not verbose and len(test_source) > 1000:
                    test_source = test_source[:1000] + "\n... (truncated)"
                lines.append(test_source)
        else:
            lines.append("Testcase: (hidden)")

        lines.append("-------------------------------------------------------------------------------------------------------------------------")
        return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description='Semantic query for function + testcase')
    parser.add_argument('--query', '-q', type=str, help='Query string (function name or code)')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results')
    parser.add_argument('--all', action='store_true', help='Include functions without tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output (no truncation)')
    parser.add_argument('--no-tests', action='store_true', help='Hide test code in results')
    parser.add_argument('--require-tests-strict', action='store_true', help='Return nothing unless tests are attached/found')
    parser.add_argument('--allow-inferred-tests', action='store_true', help='Allow inferring tests from repo when no direct tests are found')
    parser.add_argument('--infer-threshold', type=float, default=1.2, help='Min confidence to attach inferred tests')
    parser.add_argument('--embeddings-path', default='data/embeddings', help='Path to embeddings')
    args = parser.parse_args()

    if not args.query:
        args = input("Enter query: ").strip()
    if not args.query:
        return

    try:
        system = SemanticQuerySystem(
            embeddings_path=args.embeddings_path,
            infer_tests=args.allow_inferred_tests,
            infer_threshold=args.infer_threshold
        )
    except Exception as e:
        print(f"Error initializing system: {e}")
        print("Make sure you've run generate_embeddings.py first")
        return

    results = system.query(
        query=args.query,
        top_k=args.top_k,
        only_with_tests=not args.all,
        require_tests_strict=args.require_tests_strict
    )

    if results:
        for i, result in enumerate(results):
            print(system.format_result(result, i, args.verbose, show_tests=not args.no_tests))
    else:
        # intentionally quiet when no direct/inferred tests are available and only_with_tests=True
        pass


if __name__ == '__main__':
    main()
