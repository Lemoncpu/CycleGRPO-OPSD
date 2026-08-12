import importlib.util
import unittest
from pathlib import Path


_PATH = Path(__file__).parents[1] / "evaluation/refcoco/first_mask_diagnostic.py"
_SPEC = importlib.util.spec_from_file_location("first_mask_diagnostic_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
first_mask_codes = _MODULE.first_mask_codes


class FirstMaskDiagnosticTest(unittest.TestCase):
    def test_returns_first_valid_complete_group(self):
        response = (
            "<|mt_start|><|mt_0007|><|mt_0268|><|mt_end|>"
            "<|mt_start|><|mt_0008|><|mt_0269|><|mt_end|>"
        )
        self.assertEqual(first_mask_codes(response), [7, 12])

    def test_skips_invalid_group_then_uses_valid_one(self):
        response = (
            "<|mt_start|><|mt_0999|><|mt_0268|><|mt_end|>"
            "<|mt_start|><|mt_0008|><|mt_0269|><|mt_end|>"
        )
        self.assertEqual(first_mask_codes(response), [8, 13])

    def test_requires_complete_group(self):
        self.assertIsNone(first_mask_codes("<|mt_start|><|mt_0007|><|mt_0268|>"))


if __name__ == "__main__":
    unittest.main()
