# Repository List Configuration
REPO_LIST_CONFIG = {
    'repo_list_file': '../ISSTA2023-VRTesting-ReplPackage/ReplicationPackage/VR_Project_List.txt',
    'max_repos': 5,              # Maximum number of repositories to clone (for space constraints)
    'description_placeholder': 'VR Project from ISSTA2023 curated list'
}

# Tree-sitter Configuration
TREE_SITTER_CONFIG = {
    'enabled': True,                    # Enable tree-sitter parsing
    'csharp_language_path': 'tree-sitter-c-sharp',  # Path to C# language parser
    'query_timeout': 5000,              # Query timeout in milliseconds
    'max_file_size_mb': 10,             # Skip files larger than this
}


# Test File Detection
TEST_FILE_PATTERNS = [
    'test_',
    '_test.cs',
    'Tests.cs',
    'Test.cs',
    'test'
]

# File Extensions
FILE_EXTENSIONS = {
    'source': '*.cs',
    'test': '*.cs',
    'config': '*.csproj',
    'solution': '*.sln'
}

# Embedding Configuration
EMBEDDING_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',
    'batch_size': 32,
    'max_length': 512,
    'normalize': True
}

# FAISS Index Configuration
FAISS_CONFIG = {
    'index_type': 'IndexFlatL2',
    'metric': 'L2',
    'nprobe': 1
}

# Search Configuration
SEARCH_CONFIG = {
    'default_top_k': 5,
    'max_search_results': 200,
    'hybrid_weights': {
        'semantic': 0.4,
        'text': 0.4,
        'function_name': 0.2
    },
    'rerank_weights': {
        'code_similarity': 0.8,
        'rerank_score': 0.1,
        'function_name_sim': 0.1
    }
}

# Cross-Encoder Configuration
CROSS_ENCODER_CONFIG = {
    'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'max_length': 512
}

# Data Paths
DATA_PATHS = {
    'repos': 'data/repos',
    'pairs': 'data/pairs',
    'embeddings': 'data/embeddings',
    'metadata_treesitter': 'data/pairs/function_test_pairs_treesitter.json',
    'faiss_index': 'data/embeddings/faiss.index'
}

