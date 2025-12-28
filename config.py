# config.py
import argparse

def get_config():
    parser = argparse.ArgumentParser(description="AirSim DRL training config")

    # 环境相关
    parser.add_argument('--target-x', type=float, default=80.0, help='目标 x 坐标')
    parser.add_argument('--target-y', type=float, default=10.0, help='目标 y 坐标')
    parser.add_argument('--target-z', type=float, default=-5.0, help='目标 z 坐标 (负为高度)')
    parser.add_argument('--max-steps', type=int, default=300, help='每 episode 最大步数')
    parser.add_argument('--lidar-range', type=float, default=200.0, help='雷达最大量程 (m)')
    parser.add_argument('--dt', type=float, default=0.1, help='控制周期 (s)')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')

    # 训练与模型相关
    parser.add_argument('--total-timesteps', type=int, default=100000, help='训练总步数')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--policy-net', nargs='+', type=int, default=[256,256], help='policy 网络宽度列表')
    parser.add_argument('--device', type=str, default='cpu', help='训练设备, 如 cpu 或 cuda:0')

    # 保存与回调
    parser.add_argument('--save-dir', type=str, default='./models/', help='模型与检查点保存目录')
    parser.add_argument('--save-prefix', type=str, default='sac_accel', help='保存文件前缀')
    parser.add_argument('--checkpoint-freq', type=int, default=5000, help='Checkpoint 保存频率 (timesteps)')

    return parser.parse_args()
