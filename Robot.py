import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import matplotlib.transforms as transforms


class LIDAR:
    """LIDAR sensor để quét vật cản xung quanh robot"""

    def __init__(self, num_beams=36, max_range=5.0, fov=360):
        """
        num_beams: Số tia quét
        max_range: Tầm quét tối đa (m)
        fov: Góc quét (độ), 360 = quét toàn bộ
        """
        self.num_beams = num_beams
        self.max_range = max_range
        self.fov = np.radians(fov)

        # Góc các tia quét (tương đối với hướng robot)
        self.angles = np.linspace(-self.fov / 2, self.fov / 2, num_beams)

        # Kết quả quét (khoảng cách)
        self.ranges = np.full(num_beams, max_range)

        # Điểm phát hiện vật cản
        self.hit_points = []

        # Đồ họa
        self.scan_lines = []  # Các đường tia quét
        self.hit_markers = None  # Điểm phát hiện vật cản

    def scan(self, robot_x, robot_y, robot_theta, obstacles, dynamic_obstacles=None):
        """
        Thực hiện quét LIDAR
        Returns: Danh sách khoảng cách đến vật cản
        """
        self.ranges = np.full(self.num_beams, self.max_range)
        self.hit_points = []

        for i, beam_angle in enumerate(self.angles):
            # Góc tia quét trong hệ tọa độ thế giới
            world_angle = robot_theta + beam_angle

            # Tìm giao điểm gần nhất
            min_dist = self.max_range
            hit_x, hit_y = None, None

            # Kiểm tra với vật cản tĩnh
            for obs in obstacles:
                dist = self._ray_obstacle_intersection(
                    robot_x, robot_y, world_angle, obs
                )
                if dist is not None and dist < min_dist:
                    min_dist = dist
                    hit_x = robot_x + dist * np.cos(world_angle)
                    hit_y = robot_y + dist * np.sin(world_angle)

            # Kiểm tra với vật cản động (khách hàng)
            if dynamic_obstacles:
                for ox, oy, oradius in dynamic_obstacles:
                    dist = self._ray_circle_intersection(
                        robot_x, robot_y, world_angle, ox, oy, oradius
                    )
                    if dist is not None and dist < min_dist:
                        min_dist = dist
                        hit_x = robot_x + dist * np.cos(world_angle)
                        hit_y = robot_y + dist * np.sin(world_angle)

            self.ranges[i] = min_dist
            if hit_x is not None and min_dist < self.max_range:
                self.hit_points.append((hit_x, hit_y))

        return self.ranges

    def _ray_obstacle_intersection(self, rx, ry, angle, obs):
        """Tính giao điểm của tia với vật cản"""
        dx = np.cos(angle)
        dy = np.sin(angle)

        if isinstance(obs, patches.Circle):
            return self._ray_circle_intersection(
                rx, ry, angle, obs.center[0], obs.center[1], obs.radius
            )
        elif isinstance(obs, patches.Rectangle):
            return self._ray_rect_intersection(rx, ry, dx, dy, obs)
        elif hasattr(obs, "get_x") and hasattr(obs, "get_width"):
            # FancyBboxPatch (ban vuong)
            return self._ray_fancybox_intersection(rx, ry, dx, dy, obs)

        return None

    def _ray_circle_intersection(self, rx, ry, angle, cx, cy, radius):
        """Tính giao điểm tia-hình tròn"""
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Vector từ robot đến tâm hình tròn
        fx = rx - cx
        fy = ry - cy

        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        if t1 > 0:
            return t1
        if t2 > 0:
            return t2

        return None

    def _ray_rect_intersection(self, rx, ry, dx, dy, rect):
        """Tính giao điểm tia-hình chữ nhật (patches.Rectangle)"""
        try:
            x, y = rect.get_xy()
            w, h = rect.get_width(), rect.get_height()
        except:
            return None

        # 4 cạnh của hình chữ nhật
        edges = [
            ((x, y), (x + w, y)),  # Dưới
            ((x, y + h), (x + w, y + h)),  # Trên
            ((x, y), (x, y + h)),  # Trái
            ((x + w, y), (x + w, y + h)),  # Phải
        ]

        min_t = None

        for (x1, y1), (x2, y2) in edges:
            t = self._line_intersection(rx, ry, dx, dy, x1, y1, x2, y2)
            if t is not None and t > 0.1:  # Tránh self-intersection
                if min_t is None or t < min_t:
                    min_t = t

        return min_t

    def _ray_fancybox_intersection(self, rx, ry, dx, dy, fbox):
        """Tính giao điểm tia-FancyBboxPatch (bàn vuông)"""
        try:
            x = fbox.get_x()
            y = fbox.get_y()
            w = fbox.get_width()
            h = fbox.get_height()
        except:
            return None

        # 4 cạnh của hình chữ nhật
        edges = [
            ((x, y), (x + w, y)),  # Dưới
            ((x, y + h), (x + w, y + h)),  # Trên
            ((x, y), (x, y + h)),  # Trái
            ((x + w, y), (x + w, y + h)),  # Phải
        ]

        min_t = None

        for (x1, y1), (x2, y2) in edges:
            t = self._line_intersection(rx, ry, dx, dy, x1, y1, x2, y2)
            if t is not None and t > 0.1:
                if min_t is None or t < min_t:
                    min_t = t

        return min_t

    def _line_intersection(self, rx, ry, dx, dy, x1, y1, x2, y2):
        """Tính giao điểm tia với đoạn thẳng"""
        # Tia: P = (rx, ry) + t * (dx, dy)
        # Đoạn thẳng: Q = (x1, y1) + s * (x2-x1, y2-y1), 0 <= s <= 1

        denom = dx * (y2 - y1) - dy * (x2 - x1)

        if abs(denom) < 1e-10:
            return None  # Song song

        t = ((x1 - rx) * (y2 - y1) - (y1 - ry) * (x2 - x1)) / denom
        s = ((x1 - rx) * dy - (y1 - ry) * dx) / denom

        if t > 0 and 0 <= s <= 1:
            return t

        return None

    def get_scan_points(self, robot_x, robot_y, robot_theta):
        """Lấy các điểm cuối của tia quét để vẽ"""
        points = []
        for i, beam_angle in enumerate(self.angles):
            world_angle = robot_theta + beam_angle
            dist = self.ranges[i]
            end_x = robot_x + dist * np.cos(world_angle)
            end_y = robot_y + dist * np.sin(world_angle)
            points.append((end_x, end_y))
        return points


class Differential_Robot:
    def __init__(self, L, robot_length, robot_width, wheel_length, wheel_width):
        self.L = L
        self.robot_length = robot_length
        self.robot_width = robot_width
        self.wheel_length = wheel_length
        self.wheel_width = wheel_width

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v_l = 0.0
        self.v_r = 0.0
        self.v = 0.0
        self.omega = 0.0

        self.x_data = []
        self.y_data = []
        self.theta_data = []

        self.robot_body = patches.Rectangle(
            (-self.robot_length / 2, -self.robot_width / 2),
            self.robot_length,
            self.robot_width,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
        self.left_wheel = patches.Rectangle(
            (-self.wheel_length / 2, -self.wheel_width / 2),
            self.wheel_length,
            self.wheel_width,
            color="black",
        )
        self.right_wheel = patches.Rectangle(
            (-self.wheel_length / 2, -self.wheel_width / 2),
            self.wheel_length,
            self.wheel_width,
            color="black",
        )

        self.heading_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle="->", color="red", mutation_scale=15
        )
        self.left_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle="->", color="green", mutation_scale=10
        )
        self.right_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle="->", color="orange", mutation_scale=10
        )

    def update_pose(self, dt):
        self.v = (self.v_l + self.v_r) / 2
        self.omega = (self.v_r - self.v_l) / self.L
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.omega * dt
        self.theta = self.normalize_angle(self.theta)

        self.x_data.append(self.x)
        self.y_data.append(self.y)
        self.theta_data.append(self.theta)

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v_l = 0.0
        self.v_r = 0.0
        self.v = 0.0
        self.omega = 0.0
        self.x_data.clear()
        self.y_data.clear()
        self.theta_data.clear()

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def update_graphics(self, ax):
        dx = np.cos(self.theta)
        dy = np.sin(self.theta)
        perp_dx = -np.sin(self.theta)
        perp_dy = np.cos(self.theta)
        half_L = self.L / 2

        body_transform = (
            transforms.Affine2D().rotate(self.theta).translate(self.x, self.y)
            + ax.transData
        )
        self.robot_body.set_transform(body_transform)

        left_x = self.x + half_L * perp_dx
        left_y = self.y + half_L * perp_dy
        right_x = self.x - half_L * perp_dx
        right_y = self.y - half_L * perp_dy

        left_transform = (
            transforms.Affine2D().rotate(self.theta).translate(left_x, left_y)
            + ax.transData
        )
        right_transform = (
            transforms.Affine2D().rotate(self.theta).translate(right_x, right_y)
            + ax.transData
        )

        self.left_wheel.set_transform(left_transform)
        self.right_wheel.set_transform(right_transform)

        arrow_scale = 1.0
        self.heading_arrow.set_positions(
            (self.x, self.y), (self.x + arrow_scale * dx, self.y + arrow_scale * dy)
        )
        self.left_arrow.set_positions(
            (left_x, left_y),
            (
                left_x + arrow_scale * self.v_l * dx,
                left_y + arrow_scale * self.v_l * dy,
            ),
        )
        self.right_arrow.set_positions(
            (right_x, right_y),
            (
                right_x + arrow_scale * self.v_r * dx,
                right_y + arrow_scale * self.v_r * dy,
            ),
        )

    def get_robot_corners(self, x, y, theta):
        half_l = self.robot_length / 2
        half_w = self.robot_width / 2
        corners = np.array(
            [[-half_l, -half_w], [-half_l, half_w], [half_l, half_w], [half_l, -half_w]]
        )
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return (R @ corners.T).T + np.array([x, y])
