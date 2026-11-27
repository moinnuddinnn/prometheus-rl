"""Analytic examples used to validate planning before learning."""

from prometheus_rl.core import FiniteMDP, Outcome


def delayed_reward() -> FiniteMDP:
    """Cash now for 2, or wait one step to collect 5; gamma = 0.9."""
    return FiniteMDP((
        ((Outcome(1.0, 2, 2.0),), (Outcome(1.0, 1, 0.0),)),
        ((Outcome(1.0, 2, 5.0),),),
        (),
    ), initial=(1.0, 0.0, 0.0))
