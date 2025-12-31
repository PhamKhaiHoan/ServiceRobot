import numpy as np
import matplotlib.patches as patches


class BugPlanner:
    """
    Bug2 algorithm (simplified for simulation)
    Returns (v_l, v_r) for direct control like DWA
    """

    def __init__(self, robot, obstacles, step_size=0.5):
        self.robot = robot
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_speed = 3.0
        self.Kp = 2.0  # Proportional gain for steering
        self.robot_radius = max(robot.robot_length, robot.robot_width) / 2 + 0.5

        # Dynamic obstacles
        self.dynamic_obstacles = []

        # Trạng thái chờ và tránh
        self.is_blocked = False
        self.avoid_direction = 1  # 1 = left, -1 = right
        self.stuck_counter = 0  # Đếm số lần bị kẹt
        self.last_positions = []  # Lưu vị trí gần đây

    def set_dynamic_obstacles(self, obstacles_list):
        """Cập nhật danh sách vật cản động"""
        self.dynamic_obstacles = obstacles_list

    def plan(self, start, goal):
        """
        Return list of waypoints from start to goal
        """
        path = []
        current = np.array(start)
        goal = np.array(goal)

        max_iter = 500
        it = 0

        while np.linalg.norm(goal - current) > self.step_size and it < max_iter:
            it += 1

            # Step toward goal
            direction = goal - current
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            next_point = current + self.step_size * direction

            # Check collision
            if not self.check_collision(next_point):
                path.append(tuple(next_point))
                current = next_point
            else:
                # Follow obstacle boundary
                boundary_path = self.follow_boundary(current, goal)
                path.extend(boundary_path)
                if boundary_path:
                    current = np.array(boundary_path[-1])
                else:
                    break

        path.append(tuple(goal))
        return path

    def compute_velocity(self, goal):
        """
        Compute wheel velocities (v_l, v_r) to navigate toward goal
        while avoiding obstacles using Bug-like behavior
        Returns: (v_l, v_r)
        """
        robot_pos = np.array([self.robot.x, self.robot.y])
        goal_pos = np.array(goal)

        # Kiểm tra bị kẹt (không di chuyển được)
        self.last_positions.append((self.robot.x, self.robot.y))
        if len(self.last_positions) > 30:
            self.last_positions.pop(0)
            # Nếu vị trí không thay đổi nhiều -> bị kẹt
            pos_var = np.var([p[0] for p in self.last_positions]) + np.var(
                [p[1] for p in self.last_positions]
            )
            if pos_var < 0.1:
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)

        # Calculate angle to goal
        dx = goal_pos[0] - robot_pos[0]
        dy = goal_pos[1] - robot_pos[1]
        angle_to_goal = np.arctan2(dy, dx)

        # Current heading error
        angle_error = self.normalize_angle(angle_to_goal - self.robot.theta)

        # NẾU BỊ KẸT QUÁ LÂU -> LÙI VÀ XOAY
        if self.stuck_counter > 15:
            self.is_blocked = True
            v = -self.max_speed * 0.3  # Lùi
            omega = np.random.choice([-2.5, 2.5])  # Xoay ngẫu nhiên
            if self.stuck_counter > 30:
                self.stuck_counter = 0
                self.avoid_direction *= -1  # Đổi hướng tránh
        else:
            # Check for obstacles ahead
            obstacle_ahead, obstacle_side = self.check_obstacle_ahead_detailed()

            if obstacle_ahead:
                self.is_blocked = True
                # Chọn hướng tránh dựa trên vị trí vật cản
                if obstacle_side == "left":
                    self.avoid_direction = -1  # Rẽ phải
                elif obstacle_side == "right":
                    self.avoid_direction = 1  # Rẽ trái

                # Wall following behavior - tiến tới và xoay
                v = self.max_speed * 0.4
                omega = self.Kp * self.avoid_direction * 1.0
            else:
                self.is_blocked = False
                # Normal navigation toward goal
                v = self.max_speed * 0.7
                omega = self.Kp * angle_error
                omega = np.clip(omega, -3.0, 3.0)

        # Convert (v, omega) to (v_l, v_r)
        L = self.robot.L
        v_l = v - (L / 2) * omega
        v_r = v + (L / 2) * omega

        # Clip velocities
        v_l = np.clip(v_l, -self.max_speed, self.max_speed)
        v_r = np.clip(v_r, -self.max_speed, self.max_speed)

        return v_l, v_r

    def check_immediate_collision(self):
        """Kiểm tra va chạm ngay lập tức xung quanh robot"""
        check_radius = self.robot_radius + 0.2
        robot_pos = np.array([self.robot.x, self.robot.y])

        # Kiểm tra vật cản tĩnh
        if self.check_collision_at_point(robot_pos, check_radius):
            return True

        # Kiểm tra vật cản động
        for ox, oy, oradius in self.dynamic_obstacles:
            d = np.hypot(robot_pos[0] - ox, robot_pos[1] - oy) - oradius
            if d < check_radius:
                return True

        return False

    def check_obstacle_ahead_detailed(self):
        """Check if there's an obstacle ahead and determine its side"""
        look_ahead_dist = 2.0
        obstacle_detected = False
        obstacle_side = "center"

        # Kiểm tra phía trước
        for dist in np.arange(0.3, look_ahead_dist, 0.2):
            check_x = self.robot.x + dist * np.cos(self.robot.theta)
            check_y = self.robot.y + dist * np.sin(self.robot.theta)

            if self.check_collision(np.array([check_x, check_y])):
                obstacle_detected = True
                break

            # Kiểm tra vật cản động
            for ox, oy, oradius in self.dynamic_obstacles:
                d = np.hypot(check_x - ox, check_y - oy) - oradius
                if d < self.robot_radius:
                    obstacle_detected = True
                    # Xác định vật cản ở bên nào
                    angle_to_obs = np.arctan2(oy - self.robot.y, ox - self.robot.x)
                    angle_diff = self.normalize_angle(angle_to_obs - self.robot.theta)
                    if angle_diff > 0.1:
                        obstacle_side = "left"
                    elif angle_diff < -0.1:
                        obstacle_side = "right"
                    break

            if obstacle_detected:
                break

        # Kiểm tra bên trái
        if not obstacle_detected:
            for dist in np.arange(0.5, 1.5, 0.3):
                check_x = self.robot.x + dist * np.cos(self.robot.theta + 0.5)
                check_y = self.robot.y + dist * np.sin(self.robot.theta + 0.5)
                if self.check_collision(np.array([check_x, check_y])):
                    obstacle_side = "left"
                    break
                for ox, oy, oradius in self.dynamic_obstacles:
                    if (
                        np.hypot(check_x - ox, check_y - oy) - oradius
                        < self.robot_radius
                    ):
                        obstacle_side = "left"
                        break

        # Kiểm tra bên phải
        if not obstacle_detected:
            for dist in np.arange(0.5, 1.5, 0.3):
                check_x = self.robot.x + dist * np.cos(self.robot.theta - 0.5)
                check_y = self.robot.y + dist * np.sin(self.robot.theta - 0.5)
                if self.check_collision(np.array([check_x, check_y])):
                    obstacle_side = "right"
                    break
                for ox, oy, oradius in self.dynamic_obstacles:
                    if (
                        np.hypot(check_x - ox, check_y - oy) - oradius
                        < self.robot_radius
                    ):
                        obstacle_side = "right"
                        break

        return obstacle_detected, obstacle_side

    def check_obstacle_ahead(self):
        """Check if there's an obstacle directly ahead of the robot"""
        detected, _ = self.check_obstacle_ahead_detailed()
        return detected

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def check_collision_at_point(self, point, margin=0.5):
        """Kiểm tra va chạm tại một điểm với margin"""
        for obs in self.obstacles.obs:
            if isinstance(obs, patches.Circle):
                center = np.array(obs.center)
                if np.linalg.norm(point - center) <= obs.radius + margin:
                    return True
            elif isinstance(obs, patches.Rectangle):
                x, y = point
                bx, by = obs.get_xy()
                bw, bh = obs.get_width(), obs.get_height()
                if (
                    bx - margin <= x <= bx + bw + margin
                    and by - margin <= y <= by + bh + margin
                ):
                    return True
        return False

    # ====================================================
    # ================= COLLISION ========================
    # ====================================================
    def check_collision(self, point):
        for obs in self.obstacles.obs:
            if isinstance(obs, patches.Circle):
                center = np.array(obs.center)
                if np.linalg.norm(point - center) <= obs.radius + 0.5:
                    return True
            elif isinstance(obs, patches.Rectangle):
                x, y = point
                bx, by = obs.get_xy()
                bw, bh = obs.get_width(), obs.get_height()
                if bx - 0.5 <= x <= bx + bw + 0.5 and by - 0.5 <= y <= by + bh + 0.5:
                    return True
        return False

    # ====================================================
    # ============ FOLLOW OBSTACLE =======================
    # ====================================================
    def follow_boundary(self, start, goal):
        """
        Simple left-hand rule boundary following
        """
        path = []
        pos = np.array(start)

        # Find obstacle
        obstacle = self.find_nearest_obstacle(pos)
        if obstacle is None:
            return path

        # Tangent directions
        angles = np.linspace(0, 2 * np.pi, 36)

        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            next_point = pos + self.step_size * direction

            if not self.check_collision(next_point):
                # Check if moving closer to goal
                if np.linalg.norm(goal - next_point) < np.linalg.norm(goal - pos):
                    path.append(tuple(next_point))
                    break

        return path

    def find_nearest_obstacle(self, point):
        min_dist = float("inf")
        nearest = None
        for obs in self.obstacles.obs:
            if isinstance(obs, patches.Circle):
                d = np.linalg.norm(point - np.array(obs.center)) - obs.radius
            elif isinstance(obs, patches.Rectangle):
                bx, by = obs.get_xy()
                bw, bh = obs.get_width(), obs.get_height()
                cx = np.clip(point[0], bx, bx + bw)
                cy = np.clip(point[1], by, by + bh)
                d = np.linalg.norm(point - np.array([cx, cy]))
            else:
                continue

            if d < min_dist:
                min_dist = d
                nearest = obs

        return nearest
