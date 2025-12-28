# envs/airsim_drone_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import time

class AirSimDroneEnv(gym.Env):
    """
    可通过传入 config 对象进行初始化：
    env = AirSimDroneEnv(config)
    观察向量说明 (shape=21):
      [pos.x, pos.y, pos.z,
       target.x, target.y, target.z,
       vel.x, vel.y, vel.z,
       orient.w, orient.x, orient.y, orient.z,
       lidar_proximity (8 values, 0~1)]
    动作：3维加速度控制，范围 [-1,1] 对应真实加速度乘以 accel_scale
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 动作空间（加速度, 标准化到 [-1,1]）
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # 观测空间
        obs_high = np.inf * np.ones(21, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # AirSim client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # 参数
        self.target_pos = np.array([config.target_x, config.target_y, config.target_z], dtype=np.float32)
        self.max_steps = config.max_steps
        self.lidar_range = config.lidar_range
        self.dt = config.dt
        self.accel_scale = 5.0  # 动作映射到 m/s^2，可在 config 中扩展
        self.max_speed = 15.0

        # episode bookkeeping
        self.current_step = 0
        self.init_dist = 1e-6
        self.prev_dist = self.init_dist
        self.last_pos = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset sim
        try:
            self.client.reset()
        except Exception:
            # 某些配置下 reset 可能抛错，尝试继续
            pass

        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # 起飞
        self.client.takeoffAsync().join()
        # 保证在一个稳定的初始高度
        self.client.moveToPositionAsync(0, 0, -2, 3).join()
        time.sleep(0.1)

        self.current_step = 0

        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position

        # 可视化目标点
        try:
            target_vec = airsim.Vector3r(float(self.target_pos[0]), float(self.target_pos[1]), float(self.target_pos[2]))
            self.client.simPlotPoints(points=[target_vec], color_rgba=[0.0, 0.0, 1.0, 1.0], size=20.0, is_persistent=True)
        except Exception:
            pass

        # 初始化距离
        self.init_dist = np.sqrt(
            (pos.x_val - self.target_pos[0]) ** 2 +
            (pos.y_val - self.target_pos[1]) ** 2 +
            (pos.z_val - self.target_pos[2]) ** 2
        )
        self.init_dist = max(self.init_dist, 1e-6)
        self.prev_dist = self.init_dist

        # 清理轨迹并初始化上一位置
        try:
            self.client.simFlushPersistentMarkers()
        except Exception:
            pass
        self.last_pos = pos

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # 映射到真实加速度
        ax = float(np.clip(action[0], -1.0, 1.0)) * self.accel_scale
        ay = float(np.clip(action[1], -1.0, 1.0)) * self.accel_scale
        az = float(np.clip(action[2], -1.0, 1.0)) * self.accel_scale

        state = self.client.getMultirotorState()
        vel = state.kinematics_estimated.linear_velocity

        target_vx = np.clip(vel.x_val + ax * self.dt, -self.max_speed, self.max_speed)
        target_vy = np.clip(vel.y_val + ay * self.dt, -self.max_speed, self.max_speed)
        target_vz = np.clip(vel.z_val + az * self.dt, -self.max_speed, self.max_speed)

        # 发送速度指令
        try:
            self.client.moveByVelocityAsync(float(target_vx), float(target_vy), float(target_vz), self.dt).join()
        except Exception:
            # 某些情况下 join 可能失败，但仍继续
            pass

        # 绘制轨迹线（便于调试）
        new_state = self.client.getMultirotorState()
        current_pos = new_state.kinematics_estimated.position
        try:
            if self.last_pos is not None:
                self.client.simPlotLineList(
                    points=[self.last_pos, current_pos],
                    color_rgba=[1.0, 0.0, 0.0, 1.0],
                    thickness=5.0,
                    duration=0.0,
                    is_persistent=True
                )
        except Exception:
            pass
        self.last_pos = current_pos

        obs = self._get_obs()
        reward, terminated = self._compute_reward(new_state)

        self.current_step += 1
        truncated = (self.current_step >= self.max_steps)
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        state = self.client.getMultirotorState()
        kin = state.kinematics_estimated
        pos = kin.position
        vel = kin.linear_velocity
        orient = kin.orientation
        lidar_prox = self._get_lidar_proximity()

        obs = np.array([
            pos.x_val, pos.y_val, pos.z_val,
            self.target_pos[0], self.target_pos[1], self.target_pos[2],
            vel.x_val, vel.y_val, vel.z_val,
            orient.w_val, orient.x_val, orient.y_val, orient.z_val,
            *lidar_prox
        ], dtype=np.float32)
        return obs

    def _get_lidar_proximity(self):
        try:
            lidar_data = self.client.getLidarData(lidar_name="LidarSensor1")
        except Exception:
            return np.zeros(8, dtype=np.float32)

        if not hasattr(lidar_data, 'point_cloud') or lidar_data.point_cloud is None:
            return np.zeros(8, dtype=np.float32)

        point_cloud = np.array(lidar_data.point_cloud, dtype=np.float32)
        if point_cloud.size == 0:
            return np.zeros(8, dtype=np.float32)

        points = np.reshape(point_cloud, (-1, 3))

        # Z 轴过滤（只看地面以下或接近平面）
        valid_mask = points[:, 2] < 0.5
        points = points[valid_mask]
        if points.shape[0] == 0:
            return np.zeros(8, dtype=np.float32)

        local_x = points[:, 0]
        local_y = points[:, 1]
        dist_vals = np.sqrt(local_x ** 2 + local_y ** 2)
        angles = np.arctan2(local_y, local_x)
        angle_indices = ((angles + (np.pi / 8)) / (np.pi / 4)).astype(int) % 8

        dists = np.ones(8, dtype=np.float32) * self.lidar_range
        for i, d in zip(angle_indices, dist_vals):
            if d < dists[i]:
                dists[i] = d

        # 归一化并取反表示接近度（1=非常接近）
        normalized = dists / self.lidar_range
        proximity = 1.0 - normalized
        proximity = np.clip(proximity, 0.0, 1.0)
        return proximity.astype(np.float32)

    def _compute_reward(self, state):
        pos = state.kinematics_estimated.position
        curr_dist = np.sqrt(
            (pos.x_val - self.target_pos[0]) ** 2 +
            (pos.y_val - self.target_pos[1]) ** 2 +
            (pos.z_val - self.target_pos[2]) ** 2
        )

        # 进度奖励：基于距离差
        reward = (self.prev_dist - curr_dist) / max(self.init_dist, 1e-6)
        self.prev_dist = curr_dist

        # 碰撞惩罚
        try:
            collision = self.client.simGetCollisionInfo()
            if collision.has_collided:
                reward -= 10.0
                return reward, True
        except Exception:
            pass

        # 到达目标
        if curr_dist < 5.0:
            reward += 10.0
            return reward, True

        return reward, False

    def close(self):
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except Exception:
            pass

    def render(self, mode='human'):
        pass
