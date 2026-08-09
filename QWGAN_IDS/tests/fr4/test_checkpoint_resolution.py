from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cqai.synthesis import resolve_checkpoint


class ResolveCheckpointTests(unittest.TestCase):
    """Early stopping means each seed ends on a different epoch.

    A template with a fixed epoch number silently points at nothing (or worse,
    at a different seed's epoch), so a wildcard has to resolve to whichever
    checkpoint that seed actually finished on.
    """

    def _tree(self, root: Path, epochs: dict[int, list[int]]) -> None:
        for seed, seed_epochs in epochs.items():
            directory = root / f"seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            for epoch in seed_epochs:
                (directory / f"epoch{epoch:04d}.pt").write_bytes(b"x")

    def test_an_exact_path_is_returned_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._tree(root, {13: [25, 50]})
            target = root / "seed13" / "epoch0050.pt"

            self.assertEqual(resolve_checkpoint(str(target)), target)

    def test_a_wildcard_resolves_to_the_highest_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._tree(root, {13: [25, 50, 160]})

            resolved = resolve_checkpoint(str(root / "seed13" / "epoch*.pt"))
            self.assertEqual(resolved.name, "epoch0160.pt")

    def test_each_seed_resolves_to_its_own_final_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._tree(root, {13: [50, 160], 42: [50, 130], 1337: [50, 90]})

            template = str(root / "seed{seed}" / "epoch*.pt")
            for seed, expected in ((13, "epoch0160.pt"), (42, "epoch0130.pt"), (1337, "epoch0090.pt")):
                with self.subTest(seed=seed):
                    resolved = resolve_checkpoint(template.format(seed=seed))
                    self.assertEqual(resolved.name, expected)

    def test_ordering_is_numeric_not_lexicographic(self) -> None:
        """Guards the zero-padding assumption rather than relying on it."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "seed13").mkdir(parents=True)
            for name in ("epoch9.pt", "epoch10.pt", "epoch100.pt"):
                (root / "seed13" / name).write_bytes(b"x")

            resolved = resolve_checkpoint(str(root / "seed13" / "epoch*.pt"))
            self.assertEqual(resolved.name, "epoch100.pt")

    def test_a_missing_checkpoint_is_refused_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(str(root / "seed13" / "epoch*.pt"))
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(str(root / "seed13" / "epoch0050.pt"))


if __name__ == "__main__":
    unittest.main()
