"""CPU-only output-contract tests; image contents and FPS are intentionally unused."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('submission_entry', ROOT/'submission_tools/random_stage2/runtime.py')
entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)


class RandomStage2Tests(unittest.TestCase):
    def test_lengths_indices_labels_and_repeatability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = {'ONE': [7], 'SHORT': [0, 10], 'LONG': list(range(0, 6000, 3))}
            for name, indices in expected.items():
                folder = root/'stage2/images'/name; folder.mkdir(parents=True)
                for index in indices: (folder/f'frame_{index:06d}.jpg').touch()
            a = entry.predict(root, root/'no_models_needed')
            b = entry.predict(root/'stage2', root/'no_models_needed')
            self.assertTrue(a.equals(b))
            self.assertEqual(list(a.columns), ['ID','collision_frame','entry_frame','evasion_space','entry_side'])
            for row in a.itertuples():
                self.assertIn(row.entry_frame, expected[row.ID]); self.assertIn(row.collision_frame, expected[row.ID])
                self.assertLessEqual(row.entry_frame, row.collision_frame)
                self.assertIn(row.evasion_space, (0,1)); self.assertIn(row.entry_side, ('LEFT','RIGHT'))
            # Invalid FPS metadata must not affect this FPS-independent fallback.
            (root/'stage2/images/ONE/metadata.json').write_text('{"fps": 0}')
            self.assertTrue(a.equals(entry.predict(root, root)))

    def test_empty_clip_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'images/EMPTY'; folder.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'Empty'):
                entry.predict(tmp, tmp)


if __name__ == '__main__': unittest.main()
