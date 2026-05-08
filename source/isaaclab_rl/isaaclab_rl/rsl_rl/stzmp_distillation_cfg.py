# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for the ST-ZMP distillation architecture.

These mirror the pattern of :mod:`isaaclab_rl.rsl_rl.distillation_cfg` but add
all ST-ZMP specific fields (encoder hyper-parameters, joint layout, loss weights).
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .distillation_cfg import RslRlDistillationAlgorithmCfg, RslRlDistillationRunnerCfg


# ---------------------------------------------------------------------------
# Policy config
# ---------------------------------------------------------------------------


@configclass
class RslRlSTZMPStudentTeacherCfg:
    """Configuration for the :class:`~rsl_rl.modules.STZMPStudentTeacher` policy.

    Passed verbatim as ``**kwargs`` to ``STZMPStudentTeacher.__init__``.
    """

    class_name: str = "STZMPStudentTeacher"
    """Policy class name — must match the rsl_rl export."""

    # ── Action noise ────────────────────────────────────────────────────────
    init_noise_std: float = MISSING
    """Initial scalar noise std for the action distribution."""

    # ── Joint / history layout ───────────────────────────────────────────────
    history_len: int = 10
    """Number of history steps per joint (H). Default: 10 (= 200 ms at 50 Hz)."""

    num_joints: int = 12
    """Total number of robot joints."""

    num_legs: int = 4
    """Number of legs."""

    leg_joint_indices: list[list[int]] = MISSING
    """Joint index lists per leg, e.g. ``[[0,1,2],[3,4,5],[6,7,8],[9,10,11]]``
    for FL / FR / HL / HR."""

    history_order: Literal["oldest_first", "newest_first"] = "oldest_first"
    """Order of timesteps inside the flattened history vector.
    Isaac Lab uses ``"oldest_first"`` (index 0 = oldest step)."""

    # ── Per-term sizes inside obs["policy"] ─────────────────────────────────
    # Must exactly match the Isaac Lab observation group term dims and order:
    #   base_lin_vel → base_ang_vel → projected_gravity → command →
    #   joint_pos(H×N) → joint_vel(H×N) → actions(H×N) → height_scan(K)
    base_lin_vel_dim: int = 3
    base_ang_vel_dim: int = 3
    projected_gravity_dim: int = 3
    command_dim: int = 5
    # height_scan_dim is inferred automatically from the remaining policy obs dim

    # ── Encoder hyper-parameters ─────────────────────────────────────────────
    d_model: int = 32
    """Attention embedding dimension (must be divisible by num_heads)."""

    num_heads: int = 4
    """Number of cross-attention heads."""

    temporal_mlp_width: int = 64
    """Hidden width of the shared temporal MLP."""

    activation: str = "elu"
    """Activation function name (passed to rsl_rl ``resolve_nn_activation``)."""

    # ── MLP hidden dims ──────────────────────────────────────────────────────
    actor_hidden_dims: list[int] = MISSING
    """Hidden layer widths for the student actor MLP."""

    teacher_hidden_dims: list[int] = MISSING
    """Hidden layer widths for the teacher MLP."""

    # ── Action history decoupling ─────────────────────────────────────────────
    actor_action_history_steps: int = 3
    """Number of recent action steps fed to the student actor MLP.
    The encoder leg tokens always receive the full ``history_len`` steps;
    this controls only the actor's short-range action context window.
    Default: 3."""

    # ── Leg naming (for debug ONNX metadata) ─────────────────────────────────
    leg_names: list[str] = MISSING
    """Human-readable leg names in the same order as ``leg_joint_indices``.
    Used to label ``attn_weights`` outputs in the debug ONNX metadata JSON.
    Example for Spot: ``["FL", "FR", "RL", "RR"]``."""

    # ── Normalization ────────────────────────────────────────────────────────
    teacher_obs_normalization: bool = False
    """Apply empirical obs normalisation to teacher inputs."""


# ---------------------------------------------------------------------------
# Algorithm config
# ---------------------------------------------------------------------------


@configclass
class RslRlSTZMPAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """Configuration for :class:`~rsl_rl.algorithms.STZMPDistillation`.

    Extends the base distillation algorithm cfg with the NLL loss fields.
    """

    class_name: str = "STZMPDistillation"
    """Algorithm class name."""

    w_bc: float = 1.0
    """Weight for the behaviour-cloning (MSE) loss term."""

    w_nll: float = 1.0
    """Weight for the Gaussian NLL loss on the ZMP latent."""

    delta_zmp_obs_key: str = "zmp_star"
    """Key in the observation TensorDict that holds ``delta_zmp_star (Δx, Δy)``.
    Must match the corresponding entry in ``obs_groups`` of the runner config."""

    log_sigma2_load_threshold: float = 0.02
    """**Logging only — has no effect on the loss or gradients.**

    Steps where ``||delta_zmp_star||₂ >= log_sigma2_load_threshold`` are counted
    as "under load" when computing the ``sigma2_free`` / ``sigma2_load`` /
    ``sigma2_ratio`` / ``zmp_err_load`` diagnostic metrics logged to W&B.

    Set to roughly the minimum ZMP shift your force curriculum produces.
    Default 0.02 = 2 cm. Increase if your ZMP targets are smaller in scale."""


# ---------------------------------------------------------------------------
# Runner config
# ---------------------------------------------------------------------------


@configclass
class RslRlSTZMPDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Runner configuration for ST-ZMP distillation.

    Overrides the policy and algorithm type annotations so that Isaac Lab's
    config system accepts :class:`RslRlSTZMPStudentTeacherCfg` and
    :class:`RslRlSTZMPAlgorithmCfg` instead of the base distillation types.
    """

    policy: RslRlSTZMPStudentTeacherCfg = MISSING
    """ST-ZMP student-teacher policy configuration."""

    algorithm: RslRlSTZMPAlgorithmCfg = MISSING
    """ST-ZMP distillation algorithm configuration."""
