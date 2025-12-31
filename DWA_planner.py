import numpy as np
from matplotlib import patches


class DWA_Planner:
    def __init__(self, robot, obstacles, dt=0.1, predict_time=1.2):
        self.robot = robot
        self.obstacles = obstacles
        self.dt = dt
        self.predict_time = predict_time

        self.max_speed = 3.0
        self.min_speed = -0.5  # Cho phep lui
        self.max_yawrate = 3.0
        self.max_accel = 2.5
        self.max_dyawrate = 5.0

        self.v_sample_res = 0.2
        self.w_sample_res = 0.2

        self.robot_radius = (
            max(self.robot.robot_length, self.robot.robot_width) / 2 + 0.3
        )

        self.radar_range = 3.0
        self.alpha = 1.0  # Goal cost weight
        self.beta = 1.0  # Obstacle cost weight (giam de khong qua so vat can)
        self.gamma = 0.8  # Speed cost weight (TANG de uu tien di chuyen)

        self.dynamic_obstacles = []
        self.is_blocked = False
        self.stuck_counter = 0

    def set_dynamic_obstacles(self, obstacles_list):
        self.dynamic_obstacles = obstacles_list

    def plan(self, goal):
        x, y, theta = self.robot.x, self.robot.y, self.robot.theta

        # NEU DA GAN DICH -> DUNG LAI
        dist_to_goal = np.hypot(goal[0] - x, goal[1] - y)
        if dist_to_goal < 0.5:
            self.is_blocked = False
            self.stuck_counter = 0
            return 0, 0  # Dung khi gan dich

        v_current = (self.robot.v_l + self.robot.v_r) / 2.0
        omega_current = (self.robot.v_r - self.robot.v_l) / self.robot.L

        # Dynamic window mo rong
        dw = [
            max(self.min_speed, v_current - self.max_accel * self.dt),
            min(self.max_speed, v_current + self.max_accel * self.dt),
            max(-self.max_yawrate, omega_current - self.max_dyawrate * self.dt),
            min(self.max_yawrate, omega_current + self.max_dyawrate * self.dt),
        ]

        # Dam bao co du range
        if dw[1] - dw[0] < 0.5:
            dw[0] = max(self.min_speed, -0.3)
            dw[1] = min(self.max_speed, 1.0)
        if dw[3] - dw[2] < 1.0:
            dw[2] = -self.max_yawrate * 0.5
            dw[3] = self.max_yawrate * 0.5

        v_samples = np.arange(dw[0], dw[1] + 0.1, self.v_sample_res)
        w_samples = np.arange(dw[2], dw[3] + 0.1, self.w_sample_res)

        best_score = -np.inf
        best_v, best_w = 0.5, 0.0  # Mac dinh di thang voi toc do kha
        has_valid = False

        for v in v_samples:
            for w in w_samples:
                traj = self.gen_traj(x, y, theta, v, w)
                obs_cost = self.calc_obs_cost(traj)
                if obs_cost <= 0:
                    continue

                goal_cost = self.calc_goal_cost(traj, goal)

                # TANG DIEM CHO TRAJECTORY CO TOC DO TIEN
                speed_bonus = 0
                if v > 0.3:
                    speed_bonus = 0.5  # Thuong cho viec di chuyen

                # PHAT DIEM CHO XOAY TAI CHO (v nho, |w| lon)
                spin_penalty = 0
                if abs(v) < 0.2 and abs(w) > 1.0:
                    spin_penalty = -1.0  # Phat xoay tai cho

                score = (
                    self.alpha * goal_cost
                    + self.beta * obs_cost
                    + self.gamma * (v / self.max_speed)
                    + speed_bonus
                    + spin_penalty
                )

                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w
                    has_valid = True

        if not has_valid:
            self.stuck_counter += 1
            self.is_blocked = True

            # NEU GAN DICH (< 2m) MA BI KET -> DOI, KHONG CO DI VAO VAT CAN
            if dist_to_goal < 2.0:
                # Gan dich roi, chi can doi vat can di chuyen
                best_v = 0.0
                best_w = 0.0
                if self.stuck_counter > 20:
                    # Thu lui nhe neu doi qua lau
                    can_reverse = self.check_reverse_safe(x, y, theta)
                    if can_reverse:
                        best_v = -0.2
                        best_w = 1.0
                    self.stuck_counter = 0
            else:
                # Xa dich -> co thoat ket
                can_reverse = self.check_reverse_safe(x, y, theta)

                if self.stuck_counter > 10:
                    if can_reverse:
                        best_v = -0.4
                        best_w = 2.0 if self.stuck_counter % 2 == 0 else -2.0
                    else:
                        best_v = 0.2
                        best_w = 2.5 if self.stuck_counter % 2 == 0 else -2.5

                    if self.stuck_counter > 30:
                        self.stuck_counter = 0
                else:
                    angle_to_goal = np.arctan2(goal[1] - y, goal[0] - x)
                    angle_diff = self.norm_angle(angle_to_goal - theta)
                    best_w = np.sign(angle_diff) * 2.0
                    best_v = 0.15
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
            self.is_blocked = False

        v_l = best_v - (self.robot.L / 2) * best_w
        v_r = best_v + (self.robot.L / 2) * best_w
        return v_l, v_r

    def gen_traj(self, x, y, theta, v, w):
        traj = []
        for _ in range(int(self.predict_time / self.dt)):
            traj.append((x, y))
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            theta += w * self.dt
        return traj

    def calc_goal_cost(self, traj, goal):
        x, y = traj[-1]
        return 1.0 / (np.hypot(goal[0] - x, goal[1] - y) + 0.1)

    def calc_obs_cost(self, traj):
        min_d = 10.0
        rx, ry, rtheta = self.robot.x, self.robot.y, self.robot.theta

        for x, y in traj:
            # === VAT CAN TINH - KIEM TRA 360 DO DE TRANH VA CHAM ===
            for obs in self.obstacles.obs:
                d = self.dist_to_obs((x, y), obs)
                # KHONG BAO GIO CHO PHEP VA CHAM
                if d <= self.robot_radius:
                    return 0.0  # Trajectory nay KHONG HOP LE

                # Chi tinh min_d cho vat can trong radar
                obs_dist = self.dist_to_obs((rx, ry), obs)
                if obs_dist <= self.radar_range + 1.0:
                    min_d = min(min_d, d)

            # === VAT CAN DONG (KHACH) - KIEM TRA 360 DO ===
            for ox, oy, r in self.dynamic_obstacles:
                d = np.hypot(x - ox, y - oy) - r
                # KHONG BAO GIO CHO PHEP VA CHAM VOI KHACH
                if d <= self.robot_radius:
                    return 0.0  # Trajectory nay KHONG HOP LE

                # Chi tinh min_d cho khach trong radar
                dist_to_dyn = np.hypot(rx - ox, ry - oy)
                if dist_to_dyn <= self.radar_range + r:
                    min_d = min(min_d, d)

        return min(min_d, 5.0)

    def get_obs_center(self, obs):
        """Lay tam cua vat can"""
        if isinstance(obs, patches.Circle):
            return obs.center
        elif isinstance(obs, patches.Rectangle):
            bx, by = obs.get_bbox().min
            bw, bh = obs.get_width(), obs.get_height()
            return (bx + bw / 2, by + bh / 2)
        elif hasattr(obs, "get_x") and hasattr(obs, "get_width"):
            # FancyBboxPatch (ban vuong)
            bx = obs.get_x()
            by = obs.get_y()
            w, h = obs.get_width(), obs.get_height()
            return (bx + w / 2, by + h / 2)
        return None

    def dist_to_obs(self, pt, obs):
        x, y = pt
        if isinstance(obs, patches.Circle):
            return np.hypot(x - obs.center[0], y - obs.center[1]) - obs.radius
        elif isinstance(obs, patches.Rectangle):
            bx, by = obs.get_bbox().min
            bw, bh = obs.get_width(), obs.get_height()
            cx = np.clip(x, bx, bx + bw)
            cy = np.clip(y, by, by + bh)
            return np.hypot(x - cx, y - cy)
        elif hasattr(obs, "get_x") and hasattr(obs, "get_width"):
            # FancyBboxPatch (ban vuong)
            bx = obs.get_x()
            by = obs.get_y()
            bw, bh = obs.get_width(), obs.get_height()
            cx = np.clip(x, bx, bx + bw)
            cy = np.clip(y, by, by + bh)
            return np.hypot(x - cx, y - cy)
        return 10.0

    def norm_angle(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def check_reverse_safe(self, x, y, theta):
        """Kiem tra co the lui an toan khong (khong co vat can phia sau)"""
        # Diem phia sau robot (lui 1m)
        back_x = x - 1.0 * np.cos(theta)
        back_y = y - 1.0 * np.sin(theta)

        # Kiem tra vat can tinh phia sau
        for obs in self.obstacles.obs:
            d = self.dist_to_obs((back_x, back_y), obs)
            if d < self.robot_radius + 0.3:
                return False  # Co vat can phia sau

        # Kiem tra vat can dong phia sau
        for ox, oy, r in self.dynamic_obstacles:
            d = np.hypot(back_x - ox, back_y - oy) - r
            if d < self.robot_radius + 0.3:
                return False  # Co khach phia sau

        return True  # An toan de lui
