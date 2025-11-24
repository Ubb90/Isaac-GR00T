# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(self, run_name: str, exp_cfg_dir: Path | None = None):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)


class TrainingLossEarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback that monitors training loss instead of evaluation loss.
    This is more efficient as it doesn't require a separate evaluation step.
    
    Stops training when the training loss stops improving (decreasing) for a specified
    number of logging steps.
    """

    def __init__(self, patience: int = 3, threshold: float = 0.0):
        """
        Args:
            patience: Number of logging steps with no improvement after which training will be stopped.
            threshold: Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience = patience
        self.threshold = threshold
        self.best_loss = None
        self.wait = 0
        self.stopped_step = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        """Check if training should stop based on training loss improvement."""
        if logs is None:
            return control

        # Get the current training loss
        current_loss = logs.get("loss")
        if current_loss is None:
            return control

        # Initialize best_loss on first call
        if self.best_loss is None:
            self.best_loss = current_loss
            if state.is_world_process_zero:
                print(f"[Early Stopping] Initial loss: {current_loss:.6f}")
            return control

        # Check if there's an improvement
        if current_loss < self.best_loss - self.threshold:
            self.best_loss = current_loss
            self.wait = 0
            if state.is_world_process_zero:
                print(
                    f"[Early Stopping] Loss improved to {current_loss:.6f} at step {state.global_step}"
                )
        else:
            self.wait += 1
            if state.is_world_process_zero:
                print(
                    f"[Early Stopping] No improvement for {self.wait}/{self.patience} checks "
                    f"(current: {current_loss:.6f}, best: {self.best_loss:.6f})"
                )

            if self.wait >= self.patience:
                self.stopped_step = state.global_step
                control.should_training_stop = True
                if state.is_world_process_zero:
                    print(
                        f"[Early Stopping] Training stopped at step {self.stopped_step}. "
                        f"Best loss was {self.best_loss:.6f}"
                    )

        return control
