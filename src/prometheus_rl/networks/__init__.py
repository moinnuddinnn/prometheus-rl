"""Neural value approximators (install the neural extra)."""

from prometheus_rl.networks.q_network import QNetwork, fixed_target_loss, select_q

__all__ = ["QNetwork", "fixed_target_loss", "select_q"]
