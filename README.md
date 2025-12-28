# AirSim DRL Flying Car

基于 **Microsoft AirSim + Unreal Engine** 的飞行汽车（Flying Car / eVTOL）  
**深度强化学习（DRL）控制与导航实验平台**。

---

## 1. 项目简介（Introduction）

本项目旨在构建一个面向飞行汽车（Flying Car / eVTOL）的强化学习训练与评估平台.

仿真基于 AirSim 提供的高保真动力学模型，
训练算法采用 Stable-Baselines3 中的 SAC 方法。

---

## 2. 技术栈（Tech Stack）

- **Simulator**: Microsoft AirSim
- **Engine**: Unreal Engine
- **RL Framework**: Stable-Baselines3
- **Environment API**: Gymnasium
- **Language**: Python 3.9

---

## 3. 项目结构（Project Structure）

```text
.
├─ envs/
│  └─ airsim_drone_env.py    # AirSim 自定义 Gym 环境
├─ train.py                 # 训练脚本
├─ eval.py                  # 评估 
├─ config.py                # 参数配置（argparse）
├─ logs/                     # 训练日志
├─ README.md
└─ .gitignore
