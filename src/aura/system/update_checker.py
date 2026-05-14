import requests
from PyQt6.QtCore import QThread, pyqtSignal

from aura.config import GITHUB_REPOSITORY
from aura.metadata import __version__


def latest_release_api_url(repository: str = GITHUB_REPOSITORY) -> str:
    return f"https://api.github.com/repos/{repository}/releases/latest"


def parse_release_version(version: str) -> tuple[int, int, int]:
    parts = version.strip().lstrip("v").split(".")
    if len(parts) != 3:
        raise ValueError(f"Unsupported release version: {version}")
    return tuple(int(part) for part in parts)


def is_newer_release(latest_version: str, current_version: str = __version__) -> bool:
    return parse_release_version(latest_version) > parse_release_version(current_version)


class UpdateCheckerThread(QThread):
    """Check GitHub for new versions without blocking the UI."""

    found_update = pyqtSignal(str, str)

    def run(self):
        try:
            response = requests.get(latest_release_api_url(), timeout=3)
            if response.status_code == 200:
                data = response.json()
                latest_ver = data["tag_name"].lstrip("v")
                if is_newer_release(latest_ver, __version__):
                    self.found_update.emit(latest_ver, data["html_url"])
        except Exception:
            pass
