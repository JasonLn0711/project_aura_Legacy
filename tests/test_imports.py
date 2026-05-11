import importlib
import pkgutil
import unittest

import aura


class ImportSmokeTests(unittest.TestCase):
    def test_all_aura_modules_import(self):
        module_names = [
            module.name
            for module in pkgutil.walk_packages(aura.__path__, prefix=f"{aura.__name__}.")
        ]

        self.assertIn("aura.app", module_names)
        self.assertIn("aura.audio.splitter_pipeline", module_names)

        for module_name in module_names:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
