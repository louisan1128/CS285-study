"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        in_dim = state_dim

        for hidden_dims in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dims))
            layers.append(nn.ReLU())
            in_dim = hidden_dims

        layers.append(nn.Linear(in_dim, chunk_size*action_dim))
        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.sample_actions(state)
        return (pred-action_chunk).square().mean()

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        flat_actions = self.net(state)
        return flat_actions.reshape(
            state.shape[0], self.chunk_size, self.action_dim
        )


class FlowMatchingPolicy(BasePolicy):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        in_dim = state_dim + chunk_size * action_dim + 1

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        noise = torch.randn_like(action_chunk)
        t = torch.rand(batch_size, 1, 1, device=state.device)

        x = (1 - t) * noise + t * action_chunk

        inputs = torch.cat(
            [
                state,
                x.reshape(batch_size, -1),
                t.reshape(batch_size, 1),
            ],
            dim=1,
        )

        pred = self.net(inputs).reshape(
            batch_size, self.chunk_size, self.action_dim
        )
        target = action_chunk - noise

        return (pred - target).square().mean()

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        action = torch.randn(
            batch_size,
            self.chunk_size,
            self.action_dim,
            device=state.device,
        )

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full(
                (batch_size, 1),
                i / num_steps,
                device=state.device,
            )

            inputs = torch.cat(
                [state, action.reshape(batch_size, -1), t],
                dim=1,
            )

            velocity = self.net(inputs).reshape(
                batch_size, self.chunk_size, self.action_dim
            )
            action = action + dt * velocity

        return action


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
