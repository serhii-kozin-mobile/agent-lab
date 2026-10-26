"""Download the two public agent-trajectory datasets used by this study (free, no API key).
- nebius/SWE-agent-trajectories (CC-BY-4.0), shard 0 of 12 only (~89 MB)
- SWE-Gym/OpenHands-SFT-Trajectories (MIT), successful OSS split (~10 MB)
"""
from huggingface_hub import hf_hub_download

FILES = [
    ("nebius/SWE-agent-trajectories", "data/train-00000-of-00012.parquet"),
    ("SWE-Gym/OpenHands-SFT-Trajectories", "data/train.success.oss-00000-of-00001.parquet"),
]

if __name__ == "__main__":
    for repo, f in FILES:
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir=f"data/{repo.replace('/', '__')}")
        print("ok", p)
