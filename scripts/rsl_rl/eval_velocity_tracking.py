"""Velocity tracking evaluation for RSL-RL policies on K1 (Isaac Lab).

前・後・左・右 × {1.0, 0.2} m/s の計8コマンドを順に評価し、
各コマンドの追従率をターミナルに出力する。GUI でロボットの動きも確認可能。

使い方 (play.py と同じ引数で OK):
    ./isaaclab.sh -p scripts/rsl_rl/eval_velocity_tracking.py \
        --task Isaac-Velocity-Flat-K1-Play-v0 \
        --checkpoint logs/rsl_rl/k1_flat/2026-04-24_10-28-41/model_1499.pt \
        --num_envs 20

ヘッドレスで走らせたい場合は --headless を付けてください。
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports (play.py と同じ)
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Velocity tracking evaluation for RSL-RL agent.")
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
# 評価専用の引数
parser.add_argument("--settle_time", type=float, default=0.0,
                    help="コマンド印加後、計測開始までの立ち上がり時間 [s].")
parser.add_argument("--measure_time", type=float, default=10.0,
                    help="追従率を計測する時間 [s].")
# RSL-RL の CLI 引数
cli_args.add_rsl_rl_args(parser)
# AppLauncher の CLI 引数 (--headless など)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# ★ GUI をデフォルトで有効化 (--headless を付けない限り GUI で起動)
# args_cli.headless は --headless 指定時のみ True になる。何もしなければ False のまま。

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# K1 のタスク登録を取り込む (play.py と同じ)
import isaaclab_k1_soccer.tasks  # noqa: F401


# ---------------------------------------------------------------------------
# 評価コマンド: (label, vx, vy, wz)
# ---------------------------------------------------------------------------
COMMANDS: list[tuple[str, float, float, float]] = []
for speed in (1.0, 0.2):
    COMMANDS += [
        (f"前   {speed:.1f} m/s", +speed, 0.0, 0.0),
        (f"後   {speed:.1f} m/s", -speed, 0.0, 0.0),
        (f"左   {speed:.1f} m/s",  0.0, +speed, 0.0),
        (f"右   {speed:.1f} m/s",  0.0, -speed, 0.0),
    ]


def override_command(env, vx: float, vy: float, wz: float) -> None:
    """base_velocity コマンドを固定値で強制上書き (リサンプル抑止)."""
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    ref = getattr(cmd_term, "vel_command_b", None)
    if ref is None:
        ref = cmd_term.command
    device = ref.device
    num_envs = ref.shape[0]
    fixed = torch.tensor([[vx, vy, wz]], device=device).repeat(num_envs, 1)

    if hasattr(cmd_term, "vel_command_b"):
        cmd_term.vel_command_b[:] = fixed
    if hasattr(cmd_term, "command"):
        try:
            cmd_term.command[:] = fixed
        except (AttributeError, TypeError):
            pass
    if hasattr(cmd_term, "heading_target"):
        cmd_term.heading_target[:] = 0.0


def run_one_command(env, policy, policy_nn, label: str, vx: float, vy: float, wz: float,
                    settle_steps: int, measure_steps: int, dt: float) -> dict:
    """1 コマンドを走らせ、定常区間の速度ログから追従指標を返す."""
    print(f"\n▶ コマンド: {label}  (vx={vx:+.2f}, vy={vy:+.2f})")

    obs, _ = env.reset()
    if hasattr(policy_nn, "reset"):
        policy_nn.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))

    # ウォームアップ
    for _ in range(settle_steps):
        override_command(env, vx, vy, wz)
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

    # 計測
    vel_log = []
    for _ in range(measure_steps):
        override_command(env, vx, vy, wz)
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

        robot = env.unwrapped.scene["robot"]
        lin_b = robot.data.root_lin_vel_b[:, :2]   # base frame (vx, vy)
        vel_log.append(lin_b.detach().cpu().numpy())

    vels = np.stack(vel_log, axis=0)   # (T, N, 2)

    cmd_vec = np.array([vx, vy], dtype=np.float64)
    cmd_norm = np.linalg.norm(cmd_vec)
    axis = cmd_vec / cmd_norm if cmd_norm > 1e-6 else np.zeros(2)

    along = vels.reshape(-1, 2) @ axis
    mean_along = float(along.mean())
    tracking_ratio = float(mean_along / cmd_norm) if cmd_norm > 1e-6 else float("nan")

    return {
        "label": label,
        "cmd_norm": cmd_norm,
        "mean_along": mean_along,
        "tracking_ratio": tracking_ratio,
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate velocity tracking."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 評価用設定
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s,
        args_cli.settle_time + args_cli.measure_time + 1.0,
    )
    try:
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    except AttributeError:
        pass
    try:
        env_cfg.commands.base_velocity.heading_command = False
    except AttributeError:
        pass

    # checkpoint 解決
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    # 環境作成 (GUI で見るため render_mode を指定しておく)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading checkpoint: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    dt = env.unwrapped.step_dt
    settle_steps  = int(args_cli.settle_time  / dt)
    measure_steps = int(args_cli.measure_time / dt)
    print(f"[INFO] dt={dt:.4f}s, settle={settle_steps} steps, measure={measure_steps} steps")
    print(f"[INFO] GUI mode: {'OFF (headless)' if args_cli.headless else 'ON'}")

    # 評価ループ (1.0 m/s → 0.2 m/s, 方向ごと)
    results = []
    for label, vx, vy, wz in COMMANDS:
        row = run_one_command(env, policy, policy_nn, label, vx, vy, wz,
                              settle_steps, measure_steps, dt)
        results.append(row)

    # ---- 結果表示 ------------------------------------------------------
    # 方向×速度ごとに整理
    dir_names = ["前", "後", "左", "右"]
    speeds = [1.0, 0.2]
    # (speed, dir) → result のマップ
    by_key = {(round(r["cmd_norm"], 2), r["label"].split()[0]): r for r in results}

    print("\n" + "=" * 56)
    print("  速度追従率 評価結果")
    print(f"  (settle {args_cli.settle_time:.1f}s / measure {args_cli.measure_time:.1f}s, "
          f"num_envs = {env.num_envs})")
    print("=" * 56)

    for sp in speeds:
        print(f"\n  ● 指令速度 {sp:.1f} m/s")
        print(f"    {'方向':<6} {'指令':>8} {'実測':>10} {'追従率':>10}")
        print("    " + "-" * 40)
        for d in dir_names:
            r = by_key.get((round(sp, 2), d))
            if r is None:
                continue
            print(f"    {d:<6} "
                  f"{r['cmd_norm']:>7.2f}  "
                  f"{r['mean_along']:>+8.3f}   "
                  f"{r['tracking_ratio']*100:>7.1f} %")
    print("\n" + "=" * 56 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()