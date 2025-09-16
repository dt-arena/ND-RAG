import os
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import config

class RepoHarvester:
    """
    Repository harvester that clones VR repositories from a predefined list.
    It clones them to a local directory and saves metadata about the repositories.
    """
    
    def __init__(self):
        self.repos_dir = Path("data/repos")
        self.repos_dir.mkdir(parents=True, exist_ok=True)

    # Load VR repositories from predefined list file.
    def load_repo_list(self, max_repos: int = 10) -> List[Dict]:
        """Load repositories from the predefined VR project list file."""
        repo_list_file = Path(config.REPO_LIST_CONFIG['repo_list_file'])
        
        if not repo_list_file.exists():
            raise FileNotFoundError(f"Repository list file not found: {repo_list_file}")
        
        repos = []
        seen_repos = set()
        
        print(f"Loading repositories from: {repo_list_file}")
        
        with open(repo_list_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if len(repos) >= max_repos:
                    break
                    
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse GitHub URL to extract repository information
                if line.startswith('https://github.com/'):
                    try:
                        # Extract repo name from URL like: https://github.com/owner/repo.git
                        url_parts = line.replace('.git', '').split('/')
                        if len(url_parts) >= 5:
                            owner = url_parts[3]
                            repo_name = url_parts[4]
                            full_name = f"{owner}/{repo_name}"
                            
                            if full_name not in seen_repos:
                                seen_repos.add(full_name)
                                repos.append({
                                    'name': full_name,
                                    'url': line,
                                    'stars': 0,  # We don't have star count from the list
                                    'description': config.REPO_LIST_CONFIG['description_placeholder'],
                                    'language': 'C#',  # Assuming C# for VR projects
                                    'source': 'predefined_list',
                                    'line_number': line_num
                                })
                    except Exception as e:
                        print(f"Warning: Could not parse line {line_num}: {line} - {str(e)}")
                        continue
                else:
                    print(f"Warning: Skipping non-GitHub URL on line {line_num}: {line}")
        
        print(f"Loaded {len(repos)} repositories from predefined list")
        return repos


    # Clone a repository to the local directory.
    def clone_repo(self, repo: Dict) -> bool:
        repo_path = self.repos_dir / repo['name'].replace('/', '_')
        
        if repo_path.exists():
            print(f"Repository {repo['name']} already exists. Skipping...")
            return True
        
        try:
            os.system(f"git clone {repo['url']} {repo_path}")
            return True
        except Exception as e:
            print(f"Error cloning {repo['name']}: {str(e)}")
            return False

    def harvest(self, max_repos: int = None):
        """
        Main method to harvest VR repositories from the predefined list.
        
        Args:
            max_repos: Maximum number of repositories to clone. If None, uses config value.
        """
        print("Using predefined VR repository list...")
        
        # Use config max_repos if not specified
        if max_repos is None:
            max_repos = config.REPO_LIST_CONFIG['max_repos']
        
        repos = self.load_repo_list(max_repos)
        
        print(f"\nFound {len(repos)} repositories. Starting clone process...")
        successful_clones = 0
        
        for repo in tqdm(repos, desc="Cloning repositories"):
            if self.clone_repo(repo):
                successful_clones += 1
        
        print(f"\nSuccessfully cloned {successful_clones} out of {len(repos)} repositories.")
        
        # Save repository metadata
        metadata_path = self.repos_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(repos, f, indent=2)

if __name__ == "__main__":
    harvester = RepoHarvester()
    harvester.harvest() 