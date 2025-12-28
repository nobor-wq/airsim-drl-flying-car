# eval.py
import time
import numpy as np
from stable_baselines3 import SAC

from config import get_config
from envs.airsim_drone_env import AirSimDroneEnv


def make_env(cfg):
    """创建评估环境"""
    env = AirSimDroneEnv(cfg)
    return env


def run_fixed_action(env, action, num_steps=500, sleep_dt=0.0):
    """
    使用固定动作进行测试（调试动力学 / 控制方向 / 坐标系）
    """
    print("🔁 Reset 环境...")
    obs, _ = env.reset()
    time.sleep(3)

    for step in range(num_steps):
        print(f"--- Step {step + 1}/{num_steps} ---")
        print(f"执行动作: {action}")

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        print(f"reward = {reward:.4f}")

        if done:
            print("🔴 Done 信号触发，重置环境")
            obs, _ = env.reset()
            time.sleep(3)

        if sleep_dt > 0:
            time.sleep(sleep_dt)

    print("✅ 固定动作测试完成")


def run_model_policy(env, model_path, num_steps=1000, deterministic=True):
    """
    加载训练好的 SAC 模型进行评估
    """
    print(f"📦 加载模型: {model_path}")
    model = SAC.load(model_path, device=env.config.device)

    obs, _ = env.reset()
    time.sleep(3)

    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        print(f"[{step}] action={np.round(action,2)}, reward={reward:.4f}")

        if done:
            print("🔁 Episode 结束，重置环境")
            obs, _ = env.reset()
            time.sleep(2)

    print("✅ 模型评估完成")


def main():
    cfg = get_config()
    env = make_env(cfg)

    # ===============================
    # 模式 1：固定动作测试（你现在用的）
    # ===============================
    up_action = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    run_fixed_action(
        env=env,
        action=up_action,
        num_steps=500,
        sleep_dt=0.0
    )

    # ===============================
    # 模式 2：模型评估（需要时打开）
    # ===============================
    # run_model_policy(
    #     env=env,
    #     model_path=f"{cfg.save_dir}/{cfg.save_prefix}_final",
    #     num_steps=2000
    # )

    env.close()


if __name__ == "__main__":
    main()
