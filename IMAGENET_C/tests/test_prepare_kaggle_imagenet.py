import os
import tempfile
import unittest
from pathlib import Path

import torch

import sys

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import prepare_kaggle_imagenet as prep


# Tiny valid 1x1 JPEG.
TINY_JPEG = bytes(
    [
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x01, 0x00, 0x60, 0x00, 0x60, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01,
        0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03,
        0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5,
        0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04,
        0x04, 0x00, 0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05,
        0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14,
        0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1,
        0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19,
        0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38,
        0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54,
        0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84,
        0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
        0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA,
        0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4,
        0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,
        0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
        0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF,
        0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F,
        0x00, 0xD2, 0xCF, 0x20, 0xFF, 0xD9,
    ]
)


class PrepareKaggleImageNetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_jpeg(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(TINY_JPEG)

    def _write_fake_csv(self, path: Path):
        path.write_text(
            "ImageId,PredictionString\n"
            "ILSVRC2012_val_00000001,n00000001 1 1 2 2\n"
            "ILSVRC2012_val_00000002,n00000001 5 5 10 10 n00000001 1 1 2 2\n"
            "ILSVRC2012_val_00000003,n00000002 1 1 2 2\n"
            "ILSVRC2012_val_00000004,n00000002 1 1 2 2\n",
            encoding="utf-8",
        )

    def _write_fake_mapping(self, path: Path):
        path.write_text(
            "n00000001 alpha, class one\n"
            "n00000002 beta, class two\n",
            encoding="utf-8",
        )

    def test_prediction_string_multiple_boxes_same_wnid(self):
        wnid = prep.parse_prediction_string("n04447861 1 2 3 4 n04447861 5 6 7 8")
        self.assertEqual(wnid, "n04447861")

    def test_prediction_string_mixed_wnid_raises(self):
        with self.assertRaises(ValueError):
            prep.parse_prediction_string("n0001 1 2 3 4 n0002 5 6 7 8")

    def test_parse_csv_and_mapping(self):
        csv_path = self.root / "LOC_val_solution.csv"
        map_path = self.root / "LOC_synset_mapping.txt"
        self._write_fake_csv(csv_path)
        self._write_fake_mapping(map_path)

        image_to_wnid, ordered_ids = prep.parse_val_solution_csv(csv_path)
        wnid_map = prep.parse_synset_mapping(map_path)

        self.assertEqual(len(image_to_wnid), 4)
        self.assertEqual(ordered_ids[0], "ILSVRC2012_val_00000001")
        self.assertEqual(ordered_ids[-1], "ILSVRC2012_val_00000004")
        self.assertEqual(set(wnid_map.keys()), {"n00000001", "n00000002"})

    def test_missing_image_detection(self):
        image_to_wnid = {
            "ILSVRC2012_val_00000001": "n00000001",
            "ILSVRC2012_val_00000002": "n00000002",
        }
        source_images = {
            "ILSVRC2012_val_00000001": self.root / "ILSVRC2012_val_00000001.JPEG",
            "ILSVRC2012_val_00000003": self.root / "ILSVRC2012_val_00000003.JPEG",
        }
        with self.assertRaises(FileNotFoundError):
            prep.ensure_expected_counts(image_to_wnid, source_images, expected_rows=2, expected_classes=2)

    def test_invalid_wnid_detection(self):
        image_to_wnid = {
            "ILSVRC2012_val_00000001": "n00000001",
            "ILSVRC2012_val_00000002": "n99999999",
        }
        wnid_to_classes = {"n00000001": ("alpha",)}
        with self.assertRaises(ValueError):
            prep.validate_wnid_mapping(image_to_wnid, wnid_to_classes, expected_classes=1)

    def test_hardlink_creation_and_idempotent_rerun(self):
        src = self.root / "src.JPEG"
        dst = self.root / "dst.JPEG"
        self._write_jpeg(src)

        mode1 = prep.link_or_copy_file(src, dst, requested_mode="hardlink", overwrite=False)
        mode2 = prep.link_or_copy_file(src, dst, requested_mode="hardlink", overwrite=False)

        self.assertEqual(mode1, "hardlink")
        self.assertEqual(mode2, "skip")
        self.assertTrue(os.path.samefile(src, dst))

    def test_meta_bin_creation(self):
        output_root = self.root / "prepared"
        wnid_to_classes = {
            "n00000001": ("alpha",),
            "n00000002": ("beta",),
        }
        val_wnids = ["n00000001", "n00000002", "n00000001", "n00000002"]

        created = prep.write_meta_bin(output_root, wnid_to_classes, val_wnids, dry_run=False)
        self.assertTrue(created)

        loaded = torch.load(output_root / "meta.bin")
        self.assertEqual(loaded[0], wnid_to_classes)
        self.assertEqual(loaded[1], val_wnids)

    def test_prepare_synthetic_tree_and_load_torchvision_imagenet(self):
        try:
            from torchvision.datasets import ImageNet  # type: ignore
        except Exception:
            self.skipTest("torchvision not available")

        output_root = self.root / "prepared"
        val_root = output_root / "val"
        wnids = ["n00000001", "n00000002"]
        for wnid in wnids:
            (val_root / wnid).mkdir(parents=True, exist_ok=True)

        self._write_jpeg(val_root / "n00000001" / "ILSVRC2012_val_00000001.JPEG")
        self._write_jpeg(val_root / "n00000001" / "ILSVRC2012_val_00000002.JPEG")
        self._write_jpeg(val_root / "n00000002" / "ILSVRC2012_val_00000003.JPEG")
        self._write_jpeg(val_root / "n00000002" / "ILSVRC2012_val_00000004.JPEG")

        wnid_to_classes = {
            "n00000001": ("alpha",),
            "n00000002": ("beta",),
        }
        val_wnids = ["n00000001", "n00000001", "n00000002", "n00000002"]
        prep.write_meta_bin(output_root, wnid_to_classes, val_wnids, dry_run=False)

        ds = ImageNet(root=str(output_root), split="val")
        self.assertEqual(len(ds), 4)
        self.assertEqual(len(ds.wnids), 2)
        self.assertEqual(len(ds.wnid_to_idx), 2)


if __name__ == "__main__":
    unittest.main()
