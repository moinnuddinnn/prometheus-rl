"""Neural value approximators (install the neural extra)."""

from prometheus_rl.networks.q_network import (
    DuelingQNetwork,
    QNetwork,
    fixed_target_loss,
    select_q,
)

__all__ = ["DuelingQNetwork", "QNetwork", "fixed_target_loss", "select_q"]
