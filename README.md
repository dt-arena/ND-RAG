# ND-RAG: C# VR Function-Test Pair Retrieval System

A sophisticated Retrieval-Augmented Generation (RAG) system designed to find and retrieve relevant C# method-test pairs from VR-related repositories. This system is specifically optimized for Unity, Unreal Engine, and other C#-based VR development projects.

## 🎯 Purpose

This system helps VR developers by:
- Finding similar C# method implementations across projects
- Discovering existing test patterns for similar functionality
- Understanding how other developers test VR-related code
- Learning from open-source VR projects
- Researching code patterns and testing practices in the VR domain

## 🏗️ Architecture

The system follows a 5-stage pipeline:

```
GitHub VR Repos → Local Clones → Method-Test Pairs → Embeddings → FAISS Index → Query Interface
```

### 1. Repository Harvesting (`harvest_repos.py`)
- Searches GitHub for VR-related C# repositories
- Targets Unity, Unreal, Oculus, HTC Vive, SteamVR projects
- Clones repositories locally with configurable star thresholds
- Saves repository metadata for tracking

### 2. Method-Test Extraction (`extract_pairs.py`)
- Parses C# files using **tree-sitter** for accurate AST-based parsing
- Falls back to regex-based pattern matching if tree-sitter fails
- Identifies methods and their corresponding test cases
- Extracts additional metadata (line numbers, modifiers, return types)
- Creates structured data of method-test relationships

### 3. Embedding Generation (`generate_embeddings.py`)
- Uses SentenceTransformers (`all-MiniLM-L6-v2`) for vector embeddings
- Combines method and test code into single text representations
- Stores embeddings as NumPy arrays with metadata

### 4. FAISS Index Building (`build_faiss_index.py`)
- Creates a FAISS index for efficient similarity search
- Uses L2 distance metric for vector similarity
- Enables fast retrieval of similar code patterns

### 5. Query & Retrieval (`query_faiss_index.py`)
- Implements hybrid search combining semantic and text-based similarity
- Uses cross-encoder re-ranking for improved result quality
- Supports both code snippets and natural language queries

## 🚀 Features

### **Intelligent C# Code Matching**
- **Tree-sitter AST parsing** for accurate method extraction with automatic regex fallback
- **Test file detection** using C# naming conventions (`[Test]`, `Tests.cs`, `Test.cs`)
- **Rich metadata extraction** including line numbers, modifiers, and return types
- **Robust error handling** with graceful fallback mechanisms

### **Advanced Search Capabilities**
- **Semantic search** using transformer embeddings
- **Fuzzy text matching** with `thefuzz` library
- **Hybrid scoring** combining multiple similarity metrics
- **Cross-encoder re-ranking** for result quality improvement

### **VR-Specific Optimizations**
- **C# method patterns** for Unity/Unreal development
- **VR framework detection** (Oculus, HTC Vive, SteamVR)
- **Game engine integration** patterns

## 📋 Requirements

- Python 3.8+
- GitHub Personal Access Token
- Git installed locally

## 🛠️ Installation

### Quick Start (Recommended)
```bash
git clone <repository-url>
cd ND-RAG
python quick_start.py
```
This automated script will guide you through the entire setup process.

### Manual Installation
1. Clone this repository:
```bash
git clone <repository-url>
cd ND-RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
python setup_env.py
```
This interactive script will help you create the `.env` file with your GitHub token.

## 🔧 Usage

### Automated Pipeline (Recommended)
```bash
python quick_start.py
```
This runs the complete pipeline with user prompts for each step.

### Manual Step-by-Step

#### 1. Harvest VR Repositories
```bash
python harvest_repos.py
```
This will search for and clone VR-related C# repositories from GitHub.

#### 2. Extract Method-Test Pairs
```bash
python extract_pairs.py
```
This processes the cloned repositories using tree-sitter to extract method-test pairs.

#### 3. Generate Embeddings
```bash
python generate_embeddings.py
```
This creates vector embeddings for the extracted pairs.

#### 4. Build FAISS Index
```bash
python build_faiss_index.py
```
This builds the search index for fast retrieval.

#### 5. Query the System
```bash
python query_faiss_index.py --query "VR camera movement"
python query_faiss_index.py --query "public void Update()" --top_k 10
```

## 🔍 Query Examples

### Natural Language Queries
- "VR camera movement"
- "hand tracking implementation"
- "VR teleportation system"
- "Oculus controller input"

### Code Snippet Queries
- `public void Update()`
- `transform.position = Vector3.Lerp`
- `[SerializeField] private GameObject`

### Method Name Queries
- `Start`
- `Update`
- `OnTriggerEnter`
- `HandleVRInput`

## 📁 Project Structure

```
ND-RAG/
├── 📄 Core Scripts
│   ├── quick_start.py              # Main orchestrator (recommended entry point)
│   ├── setup_env.py               # Interactive environment setup
│   ├── harvest_repos.py           # Repository cloning
│   ├── extract_pairs.py           # Function-test extraction (tree-sitter)
│   ├── generate_embeddings.py     # Embedding generation
│   ├── build_faiss_index.py       # Index building
│   └── query_faiss_index.py       # Query interface
├── 🔧 Core Modules
│   ├── tree_sitter_extractor.py   # Tree-sitter based extraction
│   └── config.py                  # Configuration settings
├── 📚 Documentation
│   ├── README.md                  # This file
│   ├── MIGRATION_GUIDE.md         # Tree-sitter migration guide
│   └── PROJECT_EXECUTION_ORDER.md # Detailed execution workflow
├── ⚙️ Configuration
│   ├── requirements.txt           # Python dependencies
│   └── env_template.txt           # Environment template
└── 📊 Data
    ├── data/repos/                # Cloned repositories
    ├── data/pairs/                # Extracted function-test pairs
    └── data/embeddings/           # Vector embeddings and FAISS index
```

## ⚙️ Configuration

### Repository Search Parameters
- `min_stars`: Minimum star count for repositories (default: 10)
- `max_repos`: Maximum number of repositories to clone (default: 50)

### Search Queries
The system automatically searches for:
- Virtual Reality + C#
- VR + C#
- Augmented Reality + C#
- AR + C#
- Mixed Reality + C#
- Unity VR + C#
- Unreal VR + C#
- Oculus + C#
- HTC Vive + C#
- SteamVR + C#

## 🎮 VR Framework Support

- **Unity**: Full support for Unity C# scripts
- **Unreal Engine**: C# scripts and Blueprint equivalents
- **Oculus SDK**: Oculus-specific VR implementations
- **HTC Vive**: SteamVR and OpenVR implementations
- **SteamVR**: Valve's VR platform
- **Custom VR**: Generic VR framework support

## 🔧 Customization

### Adding New VR Frameworks
Edit `harvest_repos.py` to add new search queries:
```python
vr_queries = [
    # ... existing queries ...
    "YourFramework VR language:csharp stars:>={}".format(min_stars)
]
```

### Modifying Method Detection
The system now uses tree-sitter for accurate C# parsing. To modify extraction behavior, edit `tree_sitter_extractor.py` or `config.py` for fallback regex patterns.

## 📊 Performance

- **Repository Cloning**: ~1-5 minutes per repository (depending on size)
- **Method Extraction**: ~10-30 seconds per repository
- **Embedding Generation**: ~1-2 seconds per method-test pair
- **FAISS Index Building**: ~1-5 seconds for 1000+ pairs
- **Query Response**: <100ms for most queries

## 🐛 Troubleshooting

### Common Issues
1. **GitHub Rate Limiting**: Increase delay between API calls
2. **Large Repositories**: Some VR projects can be very large (>1GB)
3. **Encoding Issues**: Ensure C# files use UTF-8 encoding
4. **Memory Usage**: Large embedding sets may require more RAM

### Debug Mode
Enable verbose logging by modifying the scripts to include more print statements.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Enhanced tree-sitter queries for more C# constructs
- Support for more VR frameworks
- Improved test method detection
- Performance optimizations
- Additional search algorithms
- Multi-language support (Python, JavaScript, etc.)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- GitHub API for repository access
- SentenceTransformers for embeddings
- FAISS for efficient similarity search
- The VR development community for open-source projects
