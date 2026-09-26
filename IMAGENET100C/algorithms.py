"""ERM, GroupDRO, INF-TASK, and adaptive-beta IRO training objectives."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def environment_losses(model: nn.Module, minibatches, lambda_value: torch.Tensor) -> Dict[str, torch.Tensor]:
    losses = {}
    for domain, images, targets in minibatches:
        per_example_lambda = lambda_value.reshape(1, 1).expand(images.shape[0], 1)
        losses[str(domain)] = F.cross_entropy(model(images, per_example_lambda), targets)
    return losses


def threshold_cvar(losses: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Differentiable threshold-tail CVaR used by the CMNIST trainer."""
    alpha = alpha.to(device=losses.device, dtype=losses.dtype).clamp(0.0, 1.0)
    quantile = torch.quantile(losses, alpha, interpolation="linear")
    return losses[losses >= quantile].mean()


class _IcdfBeta(torch.autograd.Function):
    """CMNIST-compatible beta inverse-CDF with finite-difference gradients."""

    @staticmethod
    def forward(ctx, uniform, a, b):
        try:
            from scipy.stats import beta
        except ImportError as exc:
            raise ImportError("scipy is required for adaptive-beta IRO") from exc
        ctx.save_for_backward(uniform, a, b)
        value = beta.ppf(float(uniform.detach().cpu()), float(a.detach().cpu()), float(b.detach().cpu()))
        return uniform.new_tensor(value)

    @staticmethod
    def backward(ctx, grad_output):
        from scipy.stats import beta
        uniform, a, b = ctx.saved_tensors
        u, av, bv = (float(value.detach().cpu()) for value in (uniform, a, b))
        delta = 1e-5

        def derivative(value, function):
            low = max(delta, value - delta)
            high = value + delta
            return (function(high) - function(low)) / (high - low)

        du = derivative(u, lambda value: beta.ppf(min(1.0 - delta, value), av, bv))
        da = derivative(av, lambda value: beta.ppf(u, value, bv))
        db = derivative(bv, lambda value: beta.ppf(u, av, value))
        return grad_output * uniform.new_tensor(du), grad_output * a.new_tensor(da), grad_output * b.new_tensor(db)


class AdaptiveBetaSampler:
    """Adaptive preference distribution ported from CMNIST IRO."""

    def __init__(self, device: torch.device, learning_rate: float = 1e-6):
        self.device = device
        self.learning_rate = float(learning_rate)
        self.parameters = torch.tensor([1.0, 1.0], device=device, requires_grad=True)
        self.update_calls = 0

    @property
    def alpha_beta(self) -> Tuple[float, float]:
        values = self.parameters.detach().cpu().tolist()
        return float(values[0]), float(values[1])

    def _differentiable_samples(self, count: int) -> List[torch.Tensor]:
        uniforms = torch.rand(count, device=self.device, requires_grad=True).clamp(1e-5, 1.0 - 1e-5)
        return [_IcdfBeta.apply(value, self.parameters[0], self.parameters[1]) for value in uniforms]

    def update(self, model: nn.Module, minibatches, num_samples: int) -> Tuple[float, float]:
        model_copy = copy.deepcopy(model)
        objectives = []
        for preference in self._differentiable_samples(num_samples):
            risks = torch.stack(list(environment_losses(model_copy, minibatches, preference).values()))
            objectives.append(threshold_cvar(risks, preference))
        objective = torch.stack(objectives).mean()
        parameters = [value for value in model_copy.parameters() if value.requires_grad]
        gradients = torch.autograd.grad(objective, parameters, create_graph=True, allow_unused=True)
        norms = [gradient.norm(2) for gradient in gradients if gradient is not None]
        if norms:
            total_norm = torch.stack(norms).norm(2)
            meta_gradient = torch.autograd.grad(total_norm, self.parameters, allow_unused=True)[0]
            if meta_gradient is not None and torch.isfinite(meta_gradient).all():
                updated = (self.parameters - self.learning_rate * meta_gradient).clamp(min=1e-4)
                self.parameters = updated.detach().requires_grad_(True)
        self.update_calls += 1
        return self.alpha_beta

    def sample(self, count: int, generator: np.random.Generator) -> np.ndarray:
        a, b = self.alpha_beta
        return generator.beta(a, b, size=count)

    def state_dict(self) -> Dict[str, object]:
        a, b = self.alpha_beta
        return {"a": a, "b": b, "update_calls": self.update_calls, "learning_rate": self.learning_rate}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.parameters = torch.tensor(
            [float(state["a"]), float(state["b"])], device=self.device, requires_grad=True
        )
        self.update_calls = int(state.get("update_calls", 0))


class DomainAlgorithm:
    def __init__(
        self,
        name: str,
        model: nn.Module,
        *,
        learning_rate: float,
        weight_decay: float,
        groupdro_eta: float,
        num_lambda_samples: int,
        seed: int,
        device: torch.device,
    ):
        name = name.lower()
        if name not in {"erm", "groupdro", "inftask", "iro"}:
            raise ValueError(f"Unknown algorithm: {name}")
        if name in {"inftask", "iro"} and not getattr(model, "uses_lambda", False):
            raise TypeError(f"{name} requires a lambda-conditioned model")
        if num_lambda_samples <= 0:
            raise ValueError("num_lambda_samples must be positive")
        self.name = name
        self.model = model
        self.device = device
        self.groupdro_eta = float(groupdro_eta)
        self.num_lambda_samples = int(num_lambda_samples)
        self.numpy_rng = np.random.default_rng(seed)
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.group_weights: Dict[str, float] = {}
        self.beta_sampler = AdaptiveBetaSampler(device) if name == "iro" else None
        self.last_environment_losses: Dict[str, float] = {}
        self.last_lambdas: List[float] = []

    def _risk_vector(self, minibatches, preference: float):
        value = torch.tensor(float(preference), device=self.device)
        risks_by_domain = environment_losses(self.model, minibatches, value)
        names = list(risks_by_domain)
        return names, torch.stack([risks_by_domain[name] for name in names])

    def update(self, minibatches) -> Dict[str, object]:
        prepared = [(name, x.to(self.device), y.to(self.device)) for name, x, y in minibatches]
        if not prepared:
            raise ValueError("No active environment minibatches")
        self.model.train()
        self.optimizer.zero_grad()

        if self.name == "erm":
            all_images = torch.cat([images for _, images, _ in prepared])
            all_targets = torch.cat([targets for _, _, targets in prepared])
            preference = torch.zeros((all_images.shape[0], 1), device=self.device)
            objective = F.cross_entropy(self.model(all_images, preference), all_targets)
            names, risks = self._risk_vector(prepared, 0.0)
            self.last_lambdas = [0.0]
        elif self.name == "groupdro":
            names, risks = self._risk_vector(prepared, 0.0)
            if not self.group_weights:
                self.group_weights = {name: 1.0 / len(names) for name in names}
            weights = torch.tensor([self.group_weights[name] for name in names], device=self.device)
            weights = weights * torch.exp(self.groupdro_eta * risks.detach())
            weights = weights / weights.sum()
            self.group_weights = {name: float(value) for name, value in zip(names, weights.detach().cpu())}
            objective = torch.dot(weights, risks)
            self.last_lambdas = [0.0]
        else:
            if self.name == "iro":
                # The adaptive-Beta update differentiates through several
                # ResNet forward/backward graphs.  On small GPUs, retaining
                # all saved activations on-device can exhaust memory even
                # though the ordinary model update fits.  Offloading only
                # those saved tensors preserves the objective and effective
                # batch size; PyTorch copies them back as backward needs them.
                saved_tensor_context = (
                    torch.autograd.graph.save_on_cpu(pin_memory=True)
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with saved_tensor_context:
                    self.beta_sampler.update(self.model, prepared, self.num_lambda_samples)
                sampled = self.beta_sampler.sample(self.num_lambda_samples, self.numpy_rng)
            else:
                sampled = self.numpy_rng.beta(1.0, 1.0, size=self.num_lambda_samples)
            objectives = []
            names, risks = self._risk_vector(prepared, float(sampled[0]))
            for preference in sampled:
                current_names, current_risks = self._risk_vector(prepared, float(preference))
                if current_names != names:
                    raise RuntimeError("Environment ordering changed within an update")
                objectives.append(threshold_cvar(current_risks, current_risks.new_tensor(preference)))
            objective = torch.stack(objectives).mean()
            self.last_lambdas = [float(value) for value in sampled]

        objective.backward()
        self.optimizer.step()
        self.last_environment_losses = {
            name: float(value) for name, value in zip(names, risks.detach().cpu())
        }
        result = {
            "objective": float(objective.detach().cpu()),
            "environment_losses": dict(self.last_environment_losses),
            "lambdas": list(self.last_lambdas),
        }
        if self.beta_sampler is not None:
            result["adaptive_beta"] = self.beta_sampler.state_dict()
        if self.group_weights:
            result["group_weights"] = dict(self.group_weights)
        return result

    def checkpoint_state(self) -> Dict[str, object]:
        return {
            "algorithm": self.name,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "group_weights": self.group_weights,
            "beta_sampler": None if self.beta_sampler is None else self.beta_sampler.state_dict(),
            "num_lambda_samples": self.num_lambda_samples,
        }

