import numpy as np
from matplotlib import patches


class DWA_Planner:
    def __init__(self, robot, obstacles, dt=0.1, predict_time=2.0):
        self.robot = robot
        self.obstacles = obstacles
        self.dt = dt
        self.predict_time = predict_time

        self.max_speed = 2.0
        self.min_speed = -1.0
        self.max_yawrate = 2.0
        self.max_accel = 0.5
        self.max_dyawrate = 3.0

        self.v_sample_res = 0.1
        self.w_sample_res = 0.1

        self.robot_radius = max(self.robot.robot_length, self.robot.robot_width) / 2

        self.alpha = 0.5
        self.beta = 0.001
        self.gamma = 0.05

    def plan(self, goal):
        x = self.robot.x
        y = self.robot.y
        theta = self.robot.theta

        v_current = (self.robot.v_l + self.robot.v_r) / 2.0
        omega_current = (self.robot.v_r - self.robot.v_l) / self.robot.L

        Vs = [self.min_speed, self.max_speed, -self.max_yawrate, self.max_yawrate]
        Vd = [
            max(self.min_speed, v_current - self.max_accel * self.dt),
            min(self.max_speed, v_current + self.max_accel * self.dt),
            max(-self.max_yawrate, omega_current - self.max_dyawrate * self.dt),
            min(self.max_yawrate, omega_current + self.max_dyawrate * self.dt),
        ]

        dw = [
            max(Vs[0], Vd[0]),
            min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]),
            min(Vs[3], Vd[3]),
        ]

        v_samples = np.arange(dw[0], dw[1] + self.v_sample_res, self.v_sample_res)
        w_samples = np.arange(dw[2], dw[3] + self.w_sample_res, self.w_sample_res)

        best_score = -np.inf
        best_v = 0.0
        best_w = 0.0

        for v in v_samples:
            for w in w_samples:
                traj = self.generate_trajectory(x, y, theta, v, w)
                to_goal_cost = self.calc_to_goal_cost(traj, goal)
                dist_cost = self.calc_obstacle_cost(traj)
                speed_cost = v

                if dist_cost == 0:
                    continue

                score = (
                    self.alpha * to_goal_cost
                    + self.beta * dist_cost
                    + self.gamma * speed_cost
                )
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w

        v_l = best_v - (self.robot.L / 2) * best_w
        v_r = best_v + (self.robot.L / 2) * best_w

        v_l = np.clip(v_l, -self.max_speed, self.max_speed)
        v_r = np.clip(v_r, -self.max_speed, self.max_speed)

        return v_l, v_r

    def generate_trajectory(self, x_init, y_init, theta_init, v, w):
        traj = []
        x = x_init
        y = y_init
        theta = theta_init
        time = 0.0
        while time <= self.predict_time:
            traj.append((x, y, theta))
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            theta += w * self.dt
            theta = self.normalize_angle(theta)
            time += self.dt
        return traj

    def calc_to_goal_cost(self, traj, goal):
        x_goal, y_goal = goal
        x_end, y_end, _ = traj[-1]
        dist = np.hypot(x_goal - x_end, y_goal - y_end)
        return 1.0 / (dist + 1e-6)

    def calc_obstacle_cost(self, traj):
        min_dist = float("inf")
        for x, y, _ in traj:
            for obs in self.obstacles.obs:
                d = self.distance_to_obstacle((x, y), obs)
                if d <= self.robot_radius:
                    return 0.0
                if d < min_dist:
                    min_dist = d
        return min_dist

    def distance_to_obstacle(self, point, obs):
        p = np.array(point)
        if isinstance(obs, patches.Circle):
            center = np.array(obs.center)
            return np.linalg.norm(p - center)
        elif isinstance(obs, patches.Rectangle):
            bbox = obs.get_bbox()
            clamped_x = np.clip(p[0], bbox.xmin, bbox.xmax)
            clamped_y = np.clip(p[1], bbox.ymin, bbox.ymax)
            closest = np.array([clamped_x, clamped_y])
            return np.linalg.norm(p - closest)
        return float("inf")

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
