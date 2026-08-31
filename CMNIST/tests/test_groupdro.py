import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms import GroupDRO


class FixedFeatureNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([2.0]))

    def forward(self, x):
        return self.bias.expand(x.shape[0], 1)


def test_groupdro_updates_one_weight_per_domain():
    hparams = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "erm_pretrain_iters": 0,
        "lr_factor_reduction": 1,
        "groupdro_eta": 0.1,
    }
    algorithm = GroupDRO(FixedFeatureNetwork(), hparams, F.binary_cross_entropy_with_logits)
    batches = [
        (torch.zeros(4, 1), torch.zeros(4, 1)),
        (torch.zeros(4, 1), torch.ones(4, 1)),
    ]
    domain_losses = torch.tensor([
        F.binary_cross_entropy_with_logits(torch.full((4, 1), 2.0), batches[0][1]),
        F.binary_cross_entropy_with_logits(torch.full((4, 1), 2.0), batches[1][1]),
    ])
    expected_q = torch.softmax(0.1 * domain_losses, dim=0)

    algorithm.update(batches)

    assert algorithm.q.shape == (2,)
    assert torch.allclose(algorithm.q, expected_q, atol=1e-6)
    assert not torch.allclose(algorithm.q, torch.full((2,), 0.5))


def test_groupdro_uses_active_domain_count():
    hparams = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "erm_pretrain_iters": 0,
        "lr_factor_reduction": 1,
        "groupdro_eta": 0.1,
    }
    algorithm = GroupDRO(FixedFeatureNetwork(), hparams, F.binary_cross_entropy_with_logits)
    batches = [
        (torch.zeros(2, 1), torch.zeros(2, 1)),
        (torch.zeros(2, 1), torch.ones(2, 1)),
        (torch.zeros(2, 1), torch.ones(2, 1)),
    ]

    algorithm.update(batches)

    assert algorithm.q.shape == (3,)
    assert torch.isclose(algorithm.q.sum(), torch.tensor(1.0))
