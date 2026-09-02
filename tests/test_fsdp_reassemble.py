import importlib.util
import tempfile
import unittest
from pathlib import Path


_SCRIPT_PATH = Path(__file__).parents[1] / "tools/reassemble_fsdp_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("fsdp_reassemble_test", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


class FSDPReassembleTest(unittest.TestCase):
    def test_discovers_complete_source_world_size(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            actor_dir = Path(temporary_directory)
            for rank in range(8):
                (actor_dir / f"model_world_size_8_rank_{rank}.pt").touch()
            self.assertEqual(_MODULE.discover_checkpoint_world_size(actor_dir), 8)

    def test_rejects_missing_source_rank(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            actor_dir = Path(temporary_directory)
            for rank in (0, 1, 3):
                (actor_dir / f"model_world_size_4_rank_{rank}.pt").touch()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                _MODULE.discover_checkpoint_world_size(actor_dir)

    def test_rejects_mixed_checkpoint_world_sizes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            actor_dir = Path(temporary_directory)
            (actor_dir / "model_world_size_4_rank_0.pt").touch()
            (actor_dir / "model_world_size_8_rank_0.pt").touch()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                _MODULE.discover_checkpoint_world_size(actor_dir)


if __name__ == "__main__":
    unittest.main()
