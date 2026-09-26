import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from IMAGENET100C.algorithms import DomainAlgorithm
from IMAGENET100C.corruptions import CORRUPTION_TYPES, SEVERITIES, corrupt_image, stable_seed
from IMAGENET100C.data import ImageNet100CDataset, datasets_from_manifest, validate_label_space
from IMAGENET100C.evaluate import identification_interval, summarize, weighted_upper_cvar
from IMAGENET100C.manifests import build_manifest, load_manifest, save_manifest
from IMAGENET100C.models import build_transform_and_spec
from IMAGENET100C.train import load_config, resolve_experiment


class FakeDataset:
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        array = np.full((10, 12, 3), index % 255, dtype=np.uint8)
        return {"image": Image.fromarray(array), "label": int(self.labels[index])}


def tensor_transform(image):
    array = np.asarray(image.resize((8, 8)), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def random_backend(array, _name, _severity):
    return np.clip(array.astype(np.int16) + np.random.randint(0, 10, size=array.shape), 0, 255).astype(np.uint8)


def manifest_for(labels, counts, experiment="E0"):
    return build_manifest(
        dataset_name="fake", dataset_revision="abc123", class_names=[str(i) for i in range(100)],
        labels=labels, protocol="mechanism", experiment=experiment, condition="test",
        domain_counts=counts, seed=7, transform_spec={"test": True},
        validation_per_domain=100, weight_version="IMAGENET1K_V2",
    )


class ToyConditional(torch.nn.Module):
    uses_lambda = True

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
        self.lambda_calls = 0

    def forward(self, x, preference):
        self.lambda_calls += 1
        return self.linear(torch.cat([x, preference], dim=1))


class CoreTests(unittest.TestCase):
    def test_experiment_config_budgets_and_nested_type_sets(self):
        config = load_config(str(Path(__file__).parents[1] / "configs" / "experiments.json"))
        _, two_counts, _ = resolve_experiment(config, "E1", "", 2, 1000)
        _, four_counts, _ = resolve_experiment(config, "E1", "", 4, 1000)
        self.assertEqual(sum(two_counts.values()), 40000)
        self.assertEqual(sum(four_counts.values()), 40000)
        self.assertTrue(set(two_counts).issubset(four_counts))
        self.assertEqual(list(four_counts), config["e3b_anchor_types"])
        family_by_type = {
            corruption: family
            for family, corruptions in config["corruption_families"].items()
            for corruption in corruptions
        }
        self.assertEqual(
            {family_by_type[name] for name in four_counts},
            {"noise", "blur", "weather", "digital"},
        )
        _, missing_counts, _ = resolve_experiment(config, "E3b", "missing", 4, 1000)
        self.assertEqual(list(missing_counts.values()), [30000, 7500, 2500, 0])
        self.assertEqual(sum(missing_counts.values()), 40000)

    def test_label_range(self):
        validate_label_space(list(range(100)))
        with self.assertRaises(ValueError):
            validate_label_space([0, 100])

    def test_rgb_shape_and_deterministic_corruption(self):
        base = FakeDataset([0])
        dataset = ImageNet100CDataset(
            base, [0], tensor_transform, split="source_train", global_seed=4,
            protocol="mechanism", domain="gaussian_noise", corruption_backend=random_backend,
        )
        first, label = dataset[0]
        second, _ = dataset[0]
        self.assertEqual(label, 0)
        self.assertEqual(tuple(first.shape), (3, 8, 8))
        self.assertTrue(torch.equal(first, second))
        expected = SEVERITIES[stable_seed(4, "source_train", 0, "gaussian_noise", 0) % 5]
        self.assertEqual(dataset._condition(0)[1], expected)

    def test_exact_weight_transform_shape(self):
        transform, spec = build_transform_and_spec("IMAGENET1K_V2")
        image = Image.fromarray(np.zeros((160, 213, 3), dtype=np.uint8), mode="RGB")
        tensor = transform(image)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(spec["weights_enum"], "torchvision.models.ResNet50_Weights.IMAGENET1K_V2")

    def test_real_backend_all_15_corruptions_are_deterministic(self):
        image = Image.fromarray(np.full((160, 213, 3), 127, dtype=np.uint8), mode="RGB")
        for name in CORRUPTION_TYPES:
            first = np.asarray(corrupt_image(
                image, name, 1, global_seed=11, split="test", image_index=9, epoch=2
            ))
            second = np.asarray(corrupt_image(
                image, name, 1, global_seed=11, split="test", image_index=9, epoch=2
            ))
            self.assertEqual(first.shape, (160, 213, 3), name)
            self.assertTrue(np.array_equal(first, second), name)

    def test_manifest_repeatability_stratification_and_nonoverlap(self):
        labels = [label for label in range(100) for _ in range(20)]
        counts = {"gaussian_noise": 400, "snow": 300, "contrast": 0}
        first = manifest_for(labels, counts)
        second = manifest_for(labels, counts)
        self.assertEqual(first["source_assignments"], second["source_assignments"])
        all_groups = list(first["source_assignments"].values()) + list(first["source_validation_assignments"].values())
        flattened = [index for group in all_groups for index in group]
        self.assertEqual(len(flattened), len(set(flattened)))
        for counts_by_class in first["source_class_counts"].values():
            values = list(counts_by_class.values())
            self.assertLessEqual(max(values) - min(values), 1)
        self.assertNotIn("contrast", first["active_domain_counts"])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "manifest.json")
            save_manifest(first, path)
            self.assertEqual(load_manifest(path), first)

    def test_zero_count_environment_removed_from_datasets(self):
        labels = [label for label in range(100) for _ in range(20)]
        manifest = manifest_for(labels, {"gaussian_noise": 100, "snow": 0})
        datasets = datasets_from_manifest(FakeDataset(labels), manifest, tensor_transform)
        self.assertEqual(set(datasets), {"gaussian_noise"})

    def test_groupdro_receives_each_environment_loss(self):
        model = ToyConditional()
        algorithm = DomainAlgorithm(
            "groupdro", model, learning_rate=1e-3, weight_decay=0.0,
            groupdro_eta=0.1, num_lambda_samples=2, seed=0, device=torch.device("cpu"),
        )
        batches = [("a", torch.randn(4, 4), torch.tensor([0, 1, 2, 0])), ("b", torch.randn(3, 4), torch.tensor([1, 2, 0]))]
        result = algorithm.update(batches)
        self.assertEqual(set(result["environment_losses"]), {"a", "b"})

    def test_iro_uses_conditional_model_and_adaptive_beta(self):
        model = ToyConditional()
        algorithm = DomainAlgorithm(
            "iro", model, learning_rate=1e-3, weight_decay=0.0,
            groupdro_eta=0.1, num_lambda_samples=2, seed=0, device=torch.device("cpu"),
        )
        batches = [("a", torch.randn(3, 4), torch.tensor([0, 1, 2])), ("b", torch.randn(3, 4), torch.tensor([1, 2, 0]))]
        result = algorithm.update(batches)
        self.assertGreater(model.lambda_calls, 0)
        self.assertEqual(result["adaptive_beta"]["update_calls"], 1)
        self.assertEqual(len(result["lambdas"]), 2)
        self.assertEqual(set(result["environment_losses"]), {"a", "b"})

    def test_validation_is_declared_final_only(self):
        labels = [label for label in range(100) for _ in range(20)]
        manifest = manifest_for(labels, {"gaussian_noise": 100})
        self.assertEqual(manifest["splits"]["source"], "train")
        self.assertEqual(manifest["splits"]["final_evaluation"], "validation")
        self.assertFalse(manifest["target_validation_used_for_selection"])
        self.assertIn("source-only", manifest["selection_data"])

    def test_cvar_and_identification_interval(self):
        self.assertAlmostEqual(weighted_upper_cvar([0.0, 1.0], [0.5, 0.5], 0.0), 0.5)
        self.assertAlmostEqual(weighted_upper_cvar([0.0, 1.0], [0.5, 0.5], 0.5), 1.0)
        self.assertAlmostEqual(weighted_upper_cvar([0.0, 1.0], [0.5, 0.5], 1.0), 1.0)
        lower, upper = identification_interval([0.2, 0.4], 0.25, 0.5)
        # Lower mixture masses: .375 at .4, .375 at .2, .25 at 0;
        # its worst .5 mass is .375*.4 + .125*.2 = .175, hence CVaR=.35.
        self.assertAlmostEqual(lower, 0.35)
        self.assertAlmostEqual(upper, 0.7)

    def test_e3b_anchor_identification_uses_source_validation(self):
        labels = [label for label in range(100) for _ in range(20)]
        anchors = ["gaussian_noise", "defocus_blur", "snow", "contrast"]
        manifest = manifest_for(
            labels,
            dict(zip(anchors, [100, 100, 100, 0])),
            experiment="E3b",
        )
        condition_rows = []
        for corruption_index, corruption in enumerate(CORRUPTION_TYPES):
            for severity in SEVERITIES:
                condition_rows.append({
                    "corruption": corruption,
                    "severity": severity,
                    "top1": 0.8 - 0.01 * corruption_index,
                    "top5": 0.95,
                    "cross_entropy": 0.5,
                    "count": 10,
                })
        source_rows = [
            {"domain": anchors[0], "top1": 0.9},
            {"domain": anchors[1], "top1": 0.8},
            {"domain": anchors[2], "top1": 0.7},
        ]
        result = summarize(condition_rows, source_rows, manifest)
        interval = result["partial_identification"]["0.9"]
        self.assertAlmostEqual(interval["epsilon"], 0.25)
        self.assertEqual(interval["upper"], 1.0)
        self.assertTrue(interval["upper_endpoint_expected_vacuous"])
        self.assertEqual(result["identification_p_obs_source"], "held_out_source_validation_subset")
        self.assertEqual(len(result["per_type"]), 15)


if __name__ == "__main__":
    unittest.main()

