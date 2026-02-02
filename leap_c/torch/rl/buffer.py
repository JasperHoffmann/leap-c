import collections
import random
from collections import deque
from typing import Any, Callable, Union

import torch
from torch.utils._pytree import tree_map_only
from torch.utils.data._utils.collate import collate, default_collate_fn_map


class NStepTransitionAccumulator:
    """Accumulates transitions and computes n-step returns.

    Holds up to n transitions and emits completed n-step transitions
    when either n steps are accumulated or an episode terminates.

    When n=1, acts as a pass-through (standard 1-step TD).

    Attributes:
        n: Number of steps for n-step returns.
        gamma: Discount factor.
        transitions: Buffer holding accumulated transitions.
    """

    def __init__(self, n: int, gamma: float) -> None:
        """Initialize the n-step transition accumulator.

        Args:
            n: Number of steps for n-step returns. Use n=1 for standard 1-step TD.
            gamma: Discount factor for computing returns.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n
        self.gamma = gamma
        self.transitions: deque[tuple] = deque(maxlen=n)

    def add(self, transition: tuple) -> list[tuple]:
        """Add a transition and return any completed n-step transitions.

        Args:
            transition: Tuple of (obs, action/param, reward, obs_prime, done).

        Returns:
            List of n-step transitions ready for the replay buffer.
            Each transition has the same structure but with:
            - reward replaced by the n-step discounted return
            - obs_prime replaced by the observation n steps ahead
            - done indicating if terminal state was reached within n steps
        """
        # Short-circuit for n=1 (standard 1-step TD)
        if self.n == 1:
            return [transition]

        self.transitions.append(transition)
        _, _, _, _, done = transition

        if done:
            return self._flush_all()
        elif len(self.transitions) == self.n:
            return [self._pop_first()]
        return []

    def _compute_n_step_return(self) -> tuple[float, Any, bool]:
        """Compute discounted return from accumulated transitions.

        Returns:
            Tuple of (n_step_return, final_obs, final_done).
        """
        n_step_return = 0.0
        for i, (_, _, r, _, _) in enumerate(self.transitions):
            n_step_return += (self.gamma**i) * r

        # Get final observation and done flag
        _, _, _, final_obs, final_done = self.transitions[-1]
        return n_step_return, final_obs, final_done

    def _pop_first(self) -> tuple:
        """Pop first transition with computed n-step return."""
        n_step_return, final_obs, final_done = self._compute_n_step_return()
        obs, action, _, _, _ = self.transitions.popleft()
        return (obs, action, n_step_return, final_obs, final_done)

    def _flush_all(self) -> list[tuple]:
        """Flush all remaining transitions at episode end.

        Called when episode terminates to emit all pending transitions
        with appropriately truncated n-step returns.
        """
        results = []
        while self.transitions:
            results.append(self._pop_first())

        return results

    def reset(self) -> None:
        """Clear accumulated transitions.

        Call this when starting a new episode to discard any
        partial transitions from the previous episode.
        """
        self.transitions.clear()


def pytree_tensor_to(
    pytree: Any, device: int | str | torch.device, tensor_dtype: torch.dtype
) -> Any:
    """Convert tensors in the pytree to tensor_dtype and move them to device."""
    return tree_map_only(
        torch.Tensor,
        lambda t: t.to(device=device, dtype=tensor_dtype),
        pytree,
    )


class ReplayBuffer(torch.nn.Module):
    """Replay buffer for storing transitions.

    The replay buffer is a `deque` that stores transitions in a FIFO manner. The buffer has a
    maximum size, and when the buffer is full, the oldest transitions are discarded when appending a
    new one.

    Attributes:
        buffer: A deque that stores the transitions.
        device: The device to which all sampled tensors will be cast.
        collate_fn_map: The collate function map that informs the buffer how to form batches.
            For more information, please refer to the official pytorch documentation, e.g.,
            https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.default_collate .
        tensor_dtype: The data type to which the tensors will be cast.
    """

    buffer: collections.deque
    device: str
    collate_fn_map: dict[Union[tuple, tuple[type, ...]], Callable]
    tensor_dtype: torch.dtype

    def __init__(
        self,
        buffer_limit: int,
        device: int | str | torch.device,
        tensor_dtype: torch.dtype = torch.float32,
        collate_fn_map: dict[Union[tuple, tuple[type, ...]], Callable] | None = None,
    ) -> None:
        """Initialize the replay buffer.

        Args:
            buffer_limit: The maximum number of transitions that can be stored in the buffer.
                If the buffer is full, the oldest transition is discarded when appending a new one.
            device: The device to which all sampled tensors will be cast.
            tensor_dtype: The data type to which the sampled tensors will be cast.
            collate_fn_map: The collate function map that informs the buffer how to form batches.
                If given, extends the default collate function map of PyTorch.
        """
        super().__init__()
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = torch.device(device)
        self.tensor_dtype = tensor_dtype

        if collate_fn_map is None:
            self.collate_fn_map = default_collate_fn_map
        else:
            self.collate_fn_map = {**default_collate_fn_map, **collate_fn_map}

    def put(self, data: Any) -> None:
        """Put the data into the replay buffer. If the buffer is full, the oldest data is discarded.

        Args:
            data: The data to put into the buffer.
                It should be collatable according to the `collate` function.
        """
        self.buffer.append(data)

    def sample(self, n: int) -> Any:
        """Sample a mini-batch from the replay buffer, and collate it.

        The collate is according to the `collate` function of this class.

        Args:
            n: The number of samples to draw.
        """
        mini_batch = random.sample(self.buffer, n)
        return self.collate(mini_batch)

    def collate(self, batch: Any) -> Any:
        """Collate a batch of data according to the collate function map.

        After collating, move and cast all tensors in the
        collated batch (must be a pytree structure).

        Args:
            batch: The batch of data to collate.

        Returns:
            The collated batch.
        """
        return pytree_tensor_to(
            collate(batch, collate_fn_map=self.collate_fn_map),  # type: ignore
            device=self.device,
            tensor_dtype=self.tensor_dtype,
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def get_extra_state(self) -> dict:
        """State of the replay buffer.

        This interface is used by `state_dict` and `load_state_dict` of `nn.Module`.
        """
        return {"buffer": self.buffer}

    def set_extra_state(self, state: dict) -> None:
        """Set the state dict of the replay buffer.

        This interface is used by `state_dict` and `load_state_dict` of `nn.Module`.

        Args:
            state: The state dict to set.
        """
        buffer = state["buffer"]
        self.buffer = collections.deque(buffer, maxlen=self.buffer.maxlen)
