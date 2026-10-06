"""Download an entire Hugging Face model repository into the models folder."""

from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from urllib.parse import quote

import requests


# Edit these settings, then run: uv run python download_model.py
REPO_ID = "Qwen/Qwen3.5-4B"
REVISION = "main"
MODEL_FOLDER = "Qwen-3.5-4B-torch"
HF_TOKEN = None  # Set to a Hugging Face access token for a private/gated repository.
CHUNK_SIZE = 8 * 1024 * 1024
TIMEOUT = (20, 120)  # Connection and read timeouts in seconds.

MODELS_DIR = Path(__file__).resolve().parent / "models"


def list_repo_files(headers: dict[str, str]) -> list[tuple[str, int]]:
    url = (
        f"https://huggingface.co/api/models/{quote(REPO_ID, safe='/')}/tree/"
        f"{quote(REVISION, safe='')}"
    )
    params = {"recursive": "true"}
    files = []
    while url:
        with requests.get(url, headers=headers, params=params, timeout=TIMEOUT) as response:
            response.raise_for_status()
            for entry in response.json():
                if entry["type"] == "file":
                    path = entry["path"]
                    parts = PurePosixPath(path)
                    if parts.is_absolute() or ".." in parts.parts:
                        raise ValueError(f"Unsafe repository path: {path}")
                    files.append((path, entry["size"]))
            url = response.links.get("next", {}).get("url")
            params = None  # The next-page URL already includes its query parameters.
    return sorted(files)


def download_file(remote_name: str, expected_size: int, destination: Path, headers: dict[str, str]) -> None:
    if destination.is_file() and destination.stat().st_size == expected_size:
        print(f"Already exists, skipping: {destination}")
        return

    url = (
        f"https://huggingface.co/{quote(REPO_ID, safe='/')}/resolve/"
        f"{quote(REVISION, safe='')}/{quote(remote_name, safe='/')}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")

    with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        temp_path = None
        try:
            with NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as temp:
                temp_path = Path(temp.name)
                downloaded = 0
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        temp.write(chunk)
                        downloaded += len(chunk)
                        if expected_size:
                            print(f"\r{downloaded / 1024**2:.1f} / {expected_size / 1024**2:.1f} MiB", end="", flush=True)
            if downloaded != expected_size:
                raise OSError(f"Incomplete download: received {downloaded} of {expected_size} bytes")
            temp_path.replace(destination)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    print(f"\nSaved: {destination}")


def main() -> None:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    files = list_repo_files(headers)
    model_dir = MODELS_DIR / MODEL_FOLDER
    print(f"Found {len(files)} files in {REPO_ID} ({sum(size for _, size in files) / 1024**3:.2f} GiB)")
    for remote_name, size in files:
        download_file(remote_name, size, model_dir / remote_name, headers)


if __name__ == "__main__":
    main()
