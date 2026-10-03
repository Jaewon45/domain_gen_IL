"""Deterministic ImageNet-C corruption helpers.

``imagecorruptions`` uses process-global random number generators for several
corruptions.  We isolate and restore those states so a sample is a pure
function of its complete corruption key.
"""

from __future__ import annotations

import hashlib
import inspect
import random
import threading
from contextlib import contextmanager
from typing import Iterable

import numpy as np
from PIL import Image


CORRUPTION_TYPES = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
)
SEVERITIES = (1, 2, 3, 4, 5)
_RNG_LOCK = threading.Lock()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


@contextmanager
def isolated_random_state(seed: int):
    """Temporarily seed Python and NumPy without leaking RNG state."""
    with _RNG_LOCK:
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        np.random.seed(seed)
        random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def deterministic_choice(values: Iterable[object], *key: object):
    choices = tuple(values)
    if not choices:
        raise ValueError("Cannot choose from an empty sequence")
    return choices[stable_seed(*key) % len(choices)]


def _imagecorruptions_backend(array: np.ndarray, corruption_name: str, severity: int) -> np.ndarray:
    try:
        from imagecorruptions import corrupt
        import imagecorruptions.corruptions as corruption_functions
    except ImportError as exc:
        raise ImportError(
            "imagecorruptions is required for real ImageNet-100-C data. "
            "Install the dependencies listed in IMAGENET100C/README.md."
        ) from exc
    # imagecorruptions 1.1.2 uses the pre-scikit-image-0.20
    # ``multichannel`` keyword. Adapt only that imported function reference;
    # do not modify scikit-image globally.
    gaussian = corruption_functions.gaussian
    if "multichannel" not in inspect.signature(gaussian).parameters and not getattr(gaussian, "_imagenet100c_compat", False):
        original_gaussian = gaussian

        def gaussian_compat(image, *args, multichannel=False, **kwargs):
            if "channel_axis" not in kwargs:
                kwargs["channel_axis"] = -1 if multichannel else None
            return original_gaussian(image, *args, **kwargs)

        gaussian_compat._imagenet100c_compat = True
        corruption_functions.gaussian = gaussian_compat
    random_noise = corruption_functions.sk.util.random_noise
    if not getattr(random_noise, "_imagenet100c_deterministic", False):
        original_random_noise = random_noise

        def deterministic_random_noise(image, *args, **kwargs):
            # New scikit-image releases use an independent default_rng when no
            # rng is supplied. Derive it from our isolated, hash-seeded NumPy
            # state so impulse noise remains reproducible.
            if "rng" not in kwargs and "seed" not in kwargs:
                parameter = "rng" if "rng" in inspect.signature(original_random_noise).parameters else "seed"
                kwargs[parameter] = int(np.random.randint(0, 2**31 - 1))
            return original_random_noise(image, *args, **kwargs)

        deterministic_random_noise._imagenet100c_deterministic = True
        corruption_functions.sk.util.random_noise = deterministic_random_noise
    return corrupt(array, corruption_name=corruption_name, severity=severity)


def corrupt_image(
    image: Image.Image,
    corruption_name: str,
    severity: int,
    *,
    global_seed: int,
    split: str,
    image_index: int,
    epoch: int,
    backend=None,
) -> Image.Image:
    if corruption_name not in CORRUPTION_TYPES:
        raise ValueError(f"Unknown corruption: {corruption_name}")
    if severity not in SEVERITIES:
        raise ValueError(f"Severity must be in {SEVERITIES}, got {severity}")
    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    seed = stable_seed(global_seed, split, image_index, corruption_name, severity, epoch)
    corruption_backend = backend or _imagecorruptions_backend
    with isolated_random_state(seed):
        result = corruption_backend(array, corruption_name, severity)
    return Image.fromarray(np.asarray(result, dtype=np.uint8), mode="RGB")

