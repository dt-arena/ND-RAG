import os
import json
from pathlib import Path
from typing import List, Dict
from github import Github
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

class RepoHarvester:

    # RepoHarvester class to search and clone VR repositories from GitHub.
    # It uses the GitHub API to search for repositories related to virtual reality,
    # clones them to a local directory, and saves metadata about the repositories.
    # This class requires a GitHub token to access the GitHub API.
    # It initializes the GitHub client and sets up the local directory for repositories.
    
    # Initialize the repository harvester with GitHub token.
    def __init__(self, token: str = None):
        self.token = token or os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub token not found. Please set GITHUB_TOKEN in .env file.")
        
        self.github = Github(self.token)
        self.repos_dir = Path("data/repos")
        self.repos_dir.mkdir(parents=True, exist_ok=True)

    # Search for VR repositories on GitHub.
    def search_vr_repos(self, min_stars: int = 10, max_repos: int = 50) -> List[Dict]:
        query = "virtual reality in:description language:python stars:>={}".format(min_stars)
        results = self.github.search_repositories(query=query)
        
        repos = []
        for repo in tqdm(results[:max_repos], desc="Searching repositories"):
            repos.append({
                'name': repo.full_name,
                'url': repo.clone_url,
                'stars': repo.stargazers_count,
                'description': repo.description,
                'language': repo.language
            })
        
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

    # Main method to harvest VR repositories.
    def harvest(self, min_stars: int = 10, max_repos: int = 50):
        print("Searching for VR repositories...")
        repos = self.search_vr_repos(min_stars, max_repos)
        
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