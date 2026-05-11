import requests
from PyQt6.QtCore import QThread, pyqtSignal

from aura.metadata import __version__


class UpdateCheckerThread(QThread):
    """Check GitHub for new versions without blocking the UI."""

    found_update = pyqtSignal(str, str)

    def run(self):
        try:
            repo_url = "https://api.github.com/repos/JasonLin/UltimateAudioAssistant/releases/latest"
            response = requests.get(repo_url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                latest_ver = data["tag_name"].lstrip("v")
                current_ver = __version__.lstrip("v")
                if latest_ver > current_ver:
                    self.found_update.emit(latest_ver, data["html_url"])
        except Exception:
            pass
