import subprocess
from pathlib import Path


def clone_repository(
    repo_url: str,
    destination: Path
) -> Path:
    """
    Clone a GitHub repository to destination.

    If destination already contains the requested repository,
    reuse it. Otherwise, clone the requested repository.
    """

    destination = Path(destination)

    # If destination doesn't exist, clone directly
    if not destination.exists():

        subprocess.run(
            ["git", "clone", repo_url, str(destination)],
            check=True
        )

        return destination

    # Destination exists - verify it is a Git repository
    git_dir = destination / ".git"

    if not git_dir.exists():
        print(
            f"Destination exists but is not a Git repository: "
            f"{destination}"
        )

        # Remove/choose another destination instead of using wrong repo
        raise ValueError(
            f"Destination {destination} already exists "
            f"and is not the requested Git repository."
        )

    # Get remote URL of existing repository
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=destination,
        capture_output=True,
        text=True,
        check=True
    )

    existing_url = result.stdout.strip()

    # Normalize URLs for comparison
    existing_url = existing_url.rstrip("/").removesuffix(".git")
    requested_url = repo_url.rstrip("/").removesuffix(".git")

    if existing_url != requested_url:

        raise ValueError(
            f"Destination already contains a different repository.\n"
            f"Existing:  {existing_url}\n"
            f"Requested: {requested_url}\n"
            f"Destination: {destination}"
        )

    print(
        f"Repository already exists at {destination}. "
        f"Skipping clone."
    )

    return destination


def get_current_commit(repo_path: Path) -> str:
    """
    Get the current commit hash of the repository.
    """

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )

    return result.stdout.strip()