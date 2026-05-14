import unittest

from aura.config import GITHUB_REPOSITORY
from aura.system.update_checker import is_newer_release, latest_release_api_url, parse_release_version


class UpdateCheckerTests(unittest.TestCase):
    def test_latest_release_api_url_uses_project_repo(self):
        self.assertEqual(
            latest_release_api_url(),
            f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/latest",
        )

    def test_release_versions_compare_semantically_not_lexically(self):
        self.assertEqual(parse_release_version("v1.10.0"), (1, 10, 0))
        self.assertTrue(is_newer_release("1.10.0", "1.9.9"))
        self.assertFalse(is_newer_release("1.9.9", "1.10.0"))


if __name__ == "__main__":
    unittest.main()
