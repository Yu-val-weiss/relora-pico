"""Evaluate blimp on baseline models"""
# %pip install --quiet GitPython

import json
from pathlib import Path

import evaluate
import git
from IPython.display import clear_output


def get_step(commit: git.Commit):
    """Get commit step as int"""
    return int(commit.message.lower().split("step ")[1])


def blimp_eval(model_id: str, batch_size: int = 16):
    """Perform blimp evaluation"""
    blimp = evaluate.load("pico-lm/blimp")

    blimp_result = blimp.compute(
        model_id=model_id,
        predictions=["*"],
        batch_size=batch_size,
        trust_remote_code=True,
    )

    return blimp_result


def checkout_commit(commit: git.Commit):
    """Checks out the safetensors file at a given commit"""
    print(f"checking out model.safetensors at commit: {commit.message}")
    target_file = "model.safetensors"
    repo.git.checkout(commit.hexsha, "--", str(target_file))

    repo.git.lfs("pull")


if __name__ == "__main__":
    repo_path = Path("pico-decoder-tiny")

    # Define the repo details
    repo_url = "https://huggingface.co/pico-lm/pico-decoder-tiny"
    branch_name = "pico-decoder-tiny-1"

    # Check if the repo directory already exists
    if not repo_path.exists():
        # Clone the specific branch
        repo = git.Repo.clone_from(repo_url, repo_path, branch=branch_name)
        print(f"Cloned branch '{branch_name}' to '{repo_path}'")

        gitignore = repo_path / ".gitignore"
        with gitignore.open("w") as f:
            f.write("*\n")

    else:
        repo = git.Repo(repo_path)
        print(f"Directory '{repo_path}' already exists. Skipping clone, but checking out.")
        repo.git.checkout(branch_name)

    # evaluate!

    # note in most recent order, need to iterated reversed
    hf_commits = [commit for commit in repo.iter_commits() if "Saving HF Model" in commit.message]

    base_path = Path("blimp_results")
    base_path.mkdir(exist_ok=True)

    for commit in reversed(hf_commits):
        print(commit.message)

        step = get_step(commit)
        json_path = base_path / f"step_{step}.json"

        if json_path.exists():
            continue

        checkout_commit(commit)

        result = blimp_eval(str("." / repo_path))
        with json_path.open("w") as f:
            json.dump(result, f, indent=4, sort_keys=True)

        clear_output()
