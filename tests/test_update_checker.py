import unittest

from aura.config import GITHUB_REPOSITORY
from aura.system.update_checker import latest_release_api_url


class UpdateCheckerTests(unittest.TestCase):
    def test_latest_release_api_url_uses_project_repo(self):
        self.assertEqual(
            latest_release_api_url(),
            f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/latest",
        )


if __name__ == "__main__":
    unittest.main()
