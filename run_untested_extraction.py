#!/usr/bin/env python3
"""
Simple script to run untested function extraction with user-friendly interface.
"""

import os
import sys
from pathlib import Path
from extract_untested_functions import UntestedFunctionExtractor

def check_prerequisites():
    """Check if the required directories and files exist."""
    repos_dir = Path("data/repos")
    if not repos_dir.exists():
        print("❌ Error: data/repos directory not found!")
        print("Please run harvest_repos.py first to clone repositories.")
        return False
    
    # Check if there are any repositories
    repo_count = len([d for d in repos_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    if repo_count == 0:
        print("❌ Error: No repositories found in data/repos!")
        print("Please run harvest_repos.py first to clone repositories.")
        return False
    
    print(f"✅ Found {repo_count} repositories in data/repos")
    return True

def list_available_repos():
    """List all available repositories."""
    repos_dir = Path("data/repos")
    repos = [d.name for d in repos_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    return repos

def main():
    print("🔍 Untested Function Extractor")
    print("=" * 40)
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # List available repositories
    repos = list_available_repos()
    print(f"\nAvailable repositories:")
    for i, repo in enumerate(repos, 1):
        print(f"  {i}. {repo}")
    
    print("\nOptions:")
    print("1. Extract untested functions from ALL repositories")
    print("2. Extract untested functions from a SPECIFIC repository")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                print("\n🚀 Extracting untested functions from ALL repositories...")
                extractor = UntestedFunctionExtractor()
                extractor.extract_all_untested_functions()
                break
                
            elif choice == "2":
                print(f"\nAvailable repositories:")
                for i, repo in enumerate(repos, 1):
                    print(f"  {i}. {repo}")
                
                while True:
                    try:
                        repo_choice = input(f"\nEnter repository number (1-{len(repos)}): ").strip()
                        repo_index = int(repo_choice) - 1
                        
                        if 0 <= repo_index < len(repos):
                            selected_repo = repos[repo_index]
                            print(f"\n🚀 Extracting untested functions from {selected_repo}...")
                            extractor = UntestedFunctionExtractor()
                            extractor.extract_untested_functions_from_single_repo(selected_repo)
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(repos)}")
                    except ValueError:
                        print("Please enter a valid number")
                break
                
            elif choice == "3":
                print("👋 Goodbye!")
                break
                
            else:
                print("Please enter 1, 2, or 3")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

if __name__ == "__main__":
    main()
