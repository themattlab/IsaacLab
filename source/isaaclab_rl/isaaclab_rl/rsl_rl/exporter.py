# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os

import torch
import torch.nn as nn


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            if self.rnn_type == "lstm":
                self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self.forward_lstm
                self.reset = self.reset_memory
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
                self.reset = self.reset_memory
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward_gru(self, x):
        x = self.normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        if hasattr(self, "cell_state"):
            self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_gru(self, x_in, h_in):
        x_in = self.normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18  # was 11, but it caused problems with linux-aarch, and 18 worked well across all systems.
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=opset_version,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )


# ---------------------------------------------------------------------------
# ST-ZMP export helpers
# ---------------------------------------------------------------------------


def export_stzmp_policy_as_jit(policy: object, path: str, filename: str = "policy.pt") -> None:
    """Export an :class:`STZMPStudentTeacher` policy into a TorchScript JIT file.

    Uses ``torch.jit.trace`` so that the full encoder + student-actor pipeline
    (equivalent to ``act_inference``) is captured as a single flat
    ``forward(obs: Tensor) -> Tensor`` function.

    Args:
        policy: The ``STZMPStudentTeacher`` module returned by the runner.
        path: Directory to write the file to (created if needed).
        filename: Output filename. Defaults to ``"policy.pt"``.
    """
    module = _STZMPExportModule(policy)
    os.makedirs(path, exist_ok=True)
    module.to("cpu").eval()
    example = torch.zeros(1, module._obs_dim)
    with torch.no_grad():
        traced = torch.jit.trace(module, example)
    traced.save(os.path.join(path, filename))
    print(f"[STZMP export] JIT policy saved → {os.path.join(path, filename)}")


def export_stzmp_policy_as_onnx(
    policy: object,
    path: str,
    filename: str = "policy.onnx",
    verbose: bool = False,
) -> None:
    """Export an :class:`STZMPStudentTeacher` policy into an ONNX file.

    The exported model has a single input ``obs`` (shape ``[1, obs_dim]``)
    and a single output ``actions`` (shape ``[1, num_actions]``).

    Args:
        policy: The ``STZMPStudentTeacher`` module returned by the runner.
        path: Directory to write the file to (created if needed).
        filename: Output filename. Defaults to ``"policy.onnx"``.
        verbose: Print ONNX graph summary. Defaults to ``False``.
    """
    module = _STZMPExportModule(policy)
    os.makedirs(path, exist_ok=True)
    module.to("cpu").eval()
    example = torch.zeros(1, module._obs_dim)
    torch.onnx.export(
        module,
        example,
        os.path.join(path, filename),
        export_params=True,
        opset_version=18,
        verbose=verbose,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(f"[STZMP export] ONNX policy saved → {os.path.join(path, filename)}")


class _STZMPExportModule(nn.Module):
    """Thin, dict-free wrapper around ``STZMPStudentTeacher`` for JIT/ONNX tracing.

    ``forward(flat_obs)`` replicates ``act_inference`` without any TensorDict
    or Python dict — making it fully traceable by ``torch.jit.trace`` and
    ``torch.onnx.export``.

    The loop over legs is unrolled at trace time (``num_legs`` is a Python int),
    so the exported graph contains exactly ``num_legs`` index-gather operations.
    """

    def __init__(self, policy):
        super().__init__()
        self.encoder = copy.deepcopy(policy.encoder)
        self.student_actor = copy.deepcopy(policy.student_actor)

        # Copy per-leg joint index buffers
        self._num_legs: int = policy.num_legs
        for i in range(policy._num_leg_index_sets):
            self.register_buffer(f"_leg_idx_{i}", getattr(policy, f"_leg_idx_{i}").clone())

        # Slice bounds stored as plain Python ints (baked in at trace time)
        self._blv_s, self._blv_e = policy._sl_base_lin_vel
        self._bav_s, self._bav_e = policy._sl_base_ang_vel
        self._pg_s, self._pg_e = policy._sl_proj_gravity
        self._cmd_s, self._cmd_e = policy._sl_command
        self._jph_s, self._jph_e = policy._sl_joint_pos_hist
        self._jvh_s, self._jvh_e = policy._sl_joint_vel_hist
        self._ah_s, self._ah_e = policy._sl_action_hist
        self._hs_s, self._hs_e = policy._sl_height_scan

        self._history_len: int = policy.history_len
        self._num_joints: int = policy.num_joints
        self._newest_step_idx: int = policy.newest_step_idx

        # Total flat obs dimension (used to build the dummy tensor for export)
        self._obs_dim: int = policy._sl_height_scan[1]

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        B = flat_obs.shape[0]
        H = self._history_len
        N = self._num_joints
        idx_cur = self._newest_step_idx

        # ── Slice observation terms ──────────────────────────────────────────
        base_lin_vel = flat_obs[:, self._blv_s : self._blv_e]
        base_ang_vel = flat_obs[:, self._bav_s : self._bav_e]
        proj_gravity = flat_obs[:, self._pg_s : self._pg_e]
        command = flat_obs[:, self._cmd_s : self._cmd_e]
        joint_pos_hist = flat_obs[:, self._jph_s : self._jph_e].reshape(B, H, N)
        joint_vel_hist = flat_obs[:, self._jvh_s : self._jvh_e].reshape(B, H, N)
        action_hist = flat_obs[:, self._ah_s : self._ah_e].reshape(B, H, N)
        height_scan = flat_obs[:, self._hs_s : self._hs_e]

        # ── Build per-leg tokens (loop unrolled by tracer) ───────────────────
        leg_token_list = []
        for i in range(self._num_legs):
            idx = getattr(self, f"_leg_idx_{i}")
            q = joint_pos_hist[:, :, idx].reshape(B, -1)
            dq = joint_vel_hist[:, :, idx].reshape(B, -1)
            a = action_hist[:, :, idx].reshape(B, -1)
            leg_token_list.append(torch.cat([q, dq, a], dim=-1))
        leg_tokens = torch.stack(leg_token_list, dim=1)  # [B, num_legs, token_dim]

        # ── Base token: [projected_gravity | base_ang_vel] ───────────────────
        base_current = torch.cat([proj_gravity, base_ang_vel], dim=-1)

        # ── Encoder (deterministic — mu_zmp is used directly) ────────────────
        # need_weights=False (default) keeps the traced graph lean for deployment
        mu_zmp, logvar_zmp, _, _attn = self.encoder(leg_tokens, base_current, deterministic=True)

        # ── Current-step joint state ─────────────────────────────────────────
        joint_pos_cur = joint_pos_hist[:, idx_cur, :]
        joint_vel_cur = joint_vel_hist[:, idx_cur, :]
        actions_cur = action_hist[:, idx_cur, :]

        # ── Student actor ────────────────────────────────────────────────────
        actor_input = torch.cat(
            [
                joint_pos_cur, joint_vel_cur, actions_cur,
                base_lin_vel, base_ang_vel, proj_gravity,
                command, height_scan, mu_zmp, logvar_zmp,
            ],
            dim=-1,
        )
        return self.student_actor(actor_input)
