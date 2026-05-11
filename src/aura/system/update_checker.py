import requests
from PyQt6.QtCore import QThread, pyqtSignal

from aura.config import GITHUB_REPOSITORY
from aura.metadata import __version__


def latest_release_api_url(repository: str = GITHUB_REPOSITORY) -> str:
    return f"https://api.github.com/repos/{repository}/releases/latest"


class UpdateCheckerThread(QThread):
    """Check GitHub for new versions without blocking the UI."""

    found_update = pyqtSignal(str, str)

    def run(self):
        try:
            response = requests.get(latest_release_api_url(), timeout=3)
            if response.status_code == 200:
                data = response.json()
                latest_ver = data["tag_name"].lstrip("v")
                current_ver = __version__.lstrip("v")
                if latest_ver > current_ver:
                    self.found_update.emit(latest_ver, data["html_url"])
        except Exception:
            pass
