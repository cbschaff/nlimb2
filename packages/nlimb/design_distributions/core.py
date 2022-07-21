import torch
import torch.nn as nn
import numpy as np
from rl import CatDist, TanhNormal, Normal
import gin
from typing import Type, Sequence, Optional
from nlimb.designs import Design


class DesignDistribution(nn.Module):
    def sample(self, n: Optional[int] = None) -> Sequence[Design]:
        raise NotImplementedError

    def mode(self) -> Design:
        raise NotImplementedError

    def log_prob(self, designs: Sequence[Design]) -> torch.Tensor:
        raise NotImplementedError

    def get_log_dict(self):
        raise NotImplementedError

    def forward(self, designs: Sequence[Design]) -> torch.Tensor:
        return self.log_prob(designs)


@gin.configurable(module='nlimb')
class DescreteDesignDist(DesignDistribution):
    def __init__(self, design_class: Type[Design], n: int):
        DesignDistribution.__init__(self)
        self.design_class = design_class
        self.logits = nn.Parameter(torch.zeros((n)), requires_grad=True)

    def sample(self, n: Optional[int] = None) -> Sequence[Design]:
        dist = CatDist(self.logits)
        if n is None:
            return self.design_class.from_torch(dist.sample())
        else:
            return [self.design_class.from_torch(dist.sample()) for _ in range(n)]

    def mode(self) -> Design:
        return self.design_class.from_torch(CatDist(self.logits).mode())

    def log_prob(self, designs: Sequence[Design]) -> torch.Tensor:
        params = torch.stack([d.to_torch() for d in designs]).to(self.logits.device)
        return CatDist(self.logits).log_prob(params)

    def get_log_dict(self):
        with torch.no_grad():
            dist = CatDist(self.logits)
            data = {}
            data['entropy'] = dist.entropy().cpu().numpy()
            data['perplexity'] = np.exp(data['entropy'])
            return data


@gin.configurable(module='nlimb')
class GaussianDesignDist(DesignDistribution):
    def __init__(self, design_class: Type[Design], low: torch.Tensor, high: torch.Tensor):
        DesignDistribution.__init__(self)
        self.design_class = design_class
        self.low = nn.Parameter(low, requires_grad=False)
        self.high = nn.Parameter(high, requires_grad=False)
        self.mean = nn.Parameter(torch.zeros_like(self.low), requires_grad=True)
        self.logstd = nn.Parameter(torch.zeros_like(self.low), requires_grad=True)

    def _unnorm(self, params: torch.Tensor):
        return 0.5 * (params + 1) * (self.high - self.low) + self.low

    def _norm(self, params: torch.Tensor):
        return 2.0 * (params - self.low) / (self.high - self.low) - 1

    def sample(self, n: Optional[int] = None) -> Sequence[Design]:
        with torch.no_grad():
            dist = TanhNormal(self.mean, self.logstd.exp())

            def _sample():
                normed_s = dist.sample()
                s = self._unnorm(normed_s)
                s.pre_tanh_value = normed_s.pre_tanh_value
                return s

            if n is None:
                return self.design_class.from_torch(_sample())
            else:
                return [self.design_class.from_torch(_sample()) for _ in range(n)]

    def mode(self) -> Design:
        with torch.no_grad():
            dist = TanhNormal(self.mean, self.logstd.exp())
            return self.design_class.from_torch(self._unnorm(dist.mode()))

    def log_prob(self, designs: Sequence[Design]) -> torch.Tensor:
        params = [d.to_torch() for d in designs]
        pth = [p.pre_tanh_value for p in params]
        params = self._norm(torch.stack(params).to(self.mean.device))
        params.pre_tanh_value = torch.stack(pth).to(self.mean.device)
        return TanhNormal(self.mean, self.logstd.exp()).log_prob(params)

    def get_log_dict(self):
        with torch.no_grad():
            std = self.logstd.exp()
            data = {}
            for i, (m, s) in enumerate(zip(self.mean.cpu().numpy(), std.cpu().numpy())):
                data[f'design/mean{i:02d}'] = m
                data[f'design/std{i:02d}'] = s

            data['design/entropy'] = Normal(self.mean, std).entropy()
            return data

