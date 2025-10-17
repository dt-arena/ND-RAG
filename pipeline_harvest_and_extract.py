import os
import json
import shutil
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

import config
from harvest_repos import RepoHarvester
from tree_sitter_extractor import TreeSitterExtractor


class HarvestAndExtractPipeline:
	"""
	Clone one repository at a time, extract function-test pairs, save results, then delete the repo.
	This reduces disk usage by processing repositories sequentially.
	"""

	def __init__(self):
		self.repos_dir = Path(config.DATA_PATHS['repos'])
		self.pairs_dir = Path(config.DATA_PATHS['pairs'])
		self.pairs_dir.mkdir(parents=True, exist_ok=True)

		self.harvester = RepoHarvester()
		self.extractor = TreeSitterExtractor()

	def _global_output_path(self) -> Path:
		"""Return path for single combined JSON output."""
		return self.pairs_dir / "function_test_pairs_treesitter.json"

	def _clone_single_repo(self, repo: Dict) -> Path:
		"""Clone one repo; return local path. Raises on failure."""
		repo_path = self.repos_dir / repo['name'].replace('/', '_')
		if repo_path.exists():
			# If it already exists (from previous run), reuse it
			return repo_path
		ret = os.system(f"git clone {repo['url']} \"{repo_path}\"")
		if ret != 0:
			raise RuntimeError(f"Git clone failed for {repo['name']}")
		return repo_path

	def _delete_repo(self, repo_path: Path) -> None:
		"""Delete a cloned repository directory safely."""
		if repo_path.exists() and repo_path.is_dir():
			shutil.rmtree(repo_path, ignore_errors=True)

	def run(self, max_repos: int | None = None) -> None:
		"""Run the sequential clone → extract → save → delete pipeline."""
		if max_repos is None:
			max_repos = config.REPO_LIST_CONFIG['max_repos']

		repos: List[Dict] = self.harvester.load_repo_list(max_repos)
		if not repos:
			print("No repositories to process.")
			return

		# Track outputs and combined pairs
		metadata_path = self.pairs_dir / "processed_repos_metadata.json"
		processed_entries: List[Dict] = []
		combined_pairs: List[Dict] = []

		for repo in tqdm(repos, desc="Processing repositories one-by-one"):
			repo_path: Path | None = None
			try:
				# 1) Clone
				repo_path = self._clone_single_repo(repo)

				# 2) Extract pairs for this repo
				pairs = self.extractor.process_repository(repo_path)

				# 3) Append to global list instead of writing per-repo file
				combined_pairs.extend(pairs)

				processed_entries.append({
					"name": repo['name'],
					"url": repo['url'],
					"num_pairs": len(pairs)
				})

				print(f"Added {len(pairs)} pairs for {repo['name']} to combined output")
			except Exception as e:
				print(f"Error processing {repo.get('name', '<unknown>')}: {e}")
			finally:
				# 4) Delete repo to save disk, even if extraction failed
				if repo_path is not None:
					self._delete_repo(repo_path)

		# Write combined output and metadata summary
		global_output = self._global_output_path()
		with open(global_output, 'w', encoding='utf-8') as f:
			json.dump(combined_pairs, f, indent=2)
		print(f"\nSaved combined {len(combined_pairs)} pairs to {global_output}")

		with open(metadata_path, 'w', encoding='utf-8') as f:
			json.dump(processed_entries, f, indent=2)
		print(f"\nProcessed {len(processed_entries)} repositories. Summary: {metadata_path}")


if __name__ == "__main__":
	pipeline = HarvestAndExtractPipeline()
	pipeline.run()


