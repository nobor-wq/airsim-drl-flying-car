# train.py
import os
import numpy as np
import random
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from config import get_config
from envs.airsim_drone_env import AirSimDroneEnv

def set_seed(seed):
    import torch
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def make_env(cfg):
    env = AirSimDroneEnv(cfg)
    env = Monitor(env)
    return env

def main():
    cfg = get_config()
    set_seed(cfg.seed)

    os.makedirs(cfg.save_dir, exist_ok=True)

    env = make_env(cfg)

    policy_kwargs = dict(net_arch=list(map(int, cfg.policy_net)))

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=cfg.learning_rate,
        policy_kwargs={'net_arch': policy_kwargs['net_arch']},
        device=cfg.device
    )

    checkpoint_callback = CheckpointCallback(save_freq=cfg.checkpoint_freq, save_path=cfg.save_dir, name_prefix=cfg.save_prefix)

    print("开始训练，参数：", cfg)
    model.learn(total_timesteps=cfg.total_timesteps, callback=checkpoint_callback)
    final_path = os.path.join(cfg.save_dir, f"{cfg.save_prefix}_final")
    model.save(final_path)
    print("训练结束，模型保存到：", final_path)

    env.close()

if __name__ == "__main__":
    main()
