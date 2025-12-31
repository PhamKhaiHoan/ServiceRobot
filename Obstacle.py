import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon


class Obstacle:
    def __init__(self):
        self.obs = []  # Vật cản tĩnh
        self.dynamic_obs = []  # Vật cản động (khách hàng)

    def add(self, new_obs):
        self.obs.append(new_obs)

    def clear(self):
        self.obs.clear()

    def remove(self, obs_to_remove):
        if obs_to_remove in self.obs:
            self.obs.remove(obs_to_remove)

    def generate_random_obstacles(self, count=5, area=(-10, 10)):
        for _ in range(count):
            x, y = np.random.uniform(*area, size=2)
            w, h = np.random.uniform(1.0, 3.0, size=2)
            self.add(patches.Rectangle((x, y), w, h, color="gray"))

    def label_obstacles(self, ax):
        for i, obs in enumerate(self.obs):
            if isinstance(obs, patches.Rectangle):
                x, y = obs.get_xy()
                ax.text(x, y, f"Obs {i}", fontsize=8, color="red")
            elif isinstance(obs, patches.Circle):
                x, y = obs.center
                ax.text(x, y, f"Obs {i}", fontsize=8, color="red")


class Customer:
    """Khách hàng di chuyển thẳng liên tục, chỉ đổi hướng khi đụng bàn/tường"""

    def __init__(self, x, y, customer_id, color="orange"):
        self.x = x
        self.y = y
        self.id = customer_id
        self.radius = 0.4
        self.color = color

        # Vận tốc di chuyển - CHẬM HƠN ROBOT
        self.max_speed = 0.2  # Robot di chuyen khoang 0.5-1.0

        # Hướng di chuyển ngẫu nhiên ban đầu
        self.direction = np.random.uniform(0, 2 * np.pi)
        self.vx = self.max_speed * np.cos(self.direction)
        self.vy = self.max_speed * np.sin(self.direction)

        # Luôn đi thẳng
        self.state = "walking"
        self.target_x = x
        self.target_y = y
        self.wait_time = 0
        self.sitting_at_table = None

        # Vùng di chuyển cho phép
        self.bounds = (-12, 12, -11, 10)  # xmin, xmax, ymin, ymax

        # Đồ họa
        self.body = Circle((x, y), self.radius, color=color, alpha=0.8)
        self.head = Circle(
            (x, y + 0.2), self.radius * 0.5, color="peachpuff", alpha=0.9
        )

    def set_random_target(self, avoid_areas=None):
        """Đặt hướng đi mới ngẫu nhiên"""
        self.direction = np.random.uniform(0, 2 * np.pi)
        self.vx = self.max_speed * np.cos(self.direction)
        self.vy = self.max_speed * np.sin(self.direction)
        self.state = "walking"

    def reflect_direction(self, normal_x, normal_y):
        """Phản xạ hướng đi khi va chạm"""
        # Chuẩn hóa vector pháp tuyến
        norm = np.hypot(normal_x, normal_y)
        if norm > 0:
            normal_x /= norm
            normal_y /= norm

        # Công thức phản xạ: v' = v - 2(v.n)n
        dot = self.vx * normal_x + self.vy * normal_y
        self.vx = self.vx - 2 * dot * normal_x
        self.vy = self.vy - 2 * dot * normal_y

        # Thêm chút nhiễu để tránh bị kẹt
        self.vx += np.random.uniform(-0.05, 0.05)
        self.vy += np.random.uniform(-0.05, 0.05)

        # Chuẩn hóa lại tốc độ
        speed = np.hypot(self.vx, self.vy)
        if speed > 0:
            self.vx = (self.vx / speed) * self.max_speed
            self.vy = (self.vy / speed) * self.max_speed

        self.direction = np.arctan2(self.vy, self.vx)

    def update(
        self, dt, obstacles=None, other_customers=None, avoid_areas=None, robot=None
    ):
        """Cập nhật vị trí khách hàng - đi thẳng liên tục"""
        # Luôn di chuyển
        new_x = self.x + self.vx * dt
        new_y = self.y + self.vy * dt

        collision = False
        normal_x, normal_y = 0, 0

        # Kiểm tra va chạm với ROBOT trước
        if robot is not None:
            robot_dist = np.hypot(new_x - robot.x, new_y - robot.y)
            robot_radius = 0.8  # Bán kính an toàn của robot
            if robot_dist < self.radius + robot_radius + 0.3:
                collision = True
                normal_x = self.x - robot.x
                normal_y = self.y - robot.y
                # Đẩy ra xa robot
                if robot_dist > 0:
                    push_dist = self.radius + robot_radius + 0.5 - robot_dist
                    self.x += (normal_x / robot_dist) * push_dist * 0.5
                    self.y += (normal_y / robot_dist) * push_dist * 0.5

        # Kiểm tra va chạm với tường (bounds)
        if new_x <= self.bounds[0] + self.radius:
            collision = True
            normal_x = 1  # Phản xạ sang phải
            new_x = self.bounds[0] + self.radius + 0.1
        elif new_x >= self.bounds[1] - self.radius:
            collision = True
            normal_x = -1  # Phản xạ sang trái
            new_x = self.bounds[1] - self.radius - 0.1

        if new_y <= self.bounds[2] + self.radius:
            collision = True
            normal_y = 1  # Phản xạ lên trên
            new_y = self.bounds[2] + self.radius + 0.1
        elif new_y >= self.bounds[3] - self.radius:
            collision = True
            normal_y = -1  # Phản xạ xuống dưới
            new_y = self.bounds[3] - self.radius - 0.1

        # Kiểm tra va chạm với vật cản tĩnh (bàn)
        if not collision and obstacles:
            for obs in obstacles:
                if isinstance(obs, patches.Rectangle) or hasattr(obs, "get_xy"):
                    try:
                        ox, oy = obs.get_xy()
                        ow, oh = obs.get_width(), obs.get_height()
                        margin = self.radius + 0.2

                        if (
                            ox - margin <= new_x <= ox + ow + margin
                            and oy - margin <= new_y <= oy + oh + margin
                        ):
                            collision = True
                            # Tính vector pháp tuyến từ tâm vật cản
                            center_x = ox + ow / 2
                            center_y = oy + oh / 2
                            normal_x = self.x - center_x
                            normal_y = self.y - center_y
                            break
                    except:
                        pass
                elif isinstance(obs, Circle):
                    cx, cy = obs.center
                    r = obs.radius
                    if np.hypot(new_x - cx, new_y - cy) < r + self.radius + 0.2:
                        collision = True
                        normal_x = self.x - cx
                        normal_y = self.y - cy
                        break

        # KHÔNG kiểm tra va chạm với khách hàng khác - chỉ bàn và tường

        if collision:
            # Phản xạ hướng đi
            self.reflect_direction(normal_x, normal_y)
            # Không cập nhật vị trí khi va chạm
        else:
            # Di chuyển bình thường
            self.x = new_x
            self.y = new_y

    def update_graphics(self):
        """Cập nhật đồ họa"""
        self.body.center = (self.x, self.y)
        self.head.center = (self.x, self.y + 0.25)

    def get_patch_list(self):
        """Trả về danh sách patch để vẽ"""
        return [self.body, self.head]


class RestaurantMap:
    """Tạo bản đồ nhà hàng đẹp"""

    @staticmethod
    def create_table(x, y, table_type="square", table_name=""):
        """Tạo bàn (không có ghế)"""
        patches_list = []

        if table_type == "square":
            # Bàn vuông
            table = FancyBboxPatch(
                (x - 1.2, y - 1.2),
                2.4,
                2.4,
                boxstyle="round,pad=0.05,rounding_size=0.2",
                facecolor="saddlebrown",
                edgecolor="black",
                linewidth=2,
            )
            patches_list.append(table)

        elif table_type == "round":
            # Bàn tròn
            table = Circle(
                (x, y), 1.2, facecolor="saddlebrown", edgecolor="black", linewidth=2
            )
            patches_list.append(table)

        return patches_list

    @staticmethod
    def create_kitchen(x, y, width=8, height=3):
        """Tạo khu vực bếp"""
        patches_list = []

        # Quầy bếp chính
        counter = FancyBboxPatch(
            (x - width / 2, y),
            width,
            height,
            boxstyle="round,pad=0.05,rounding_size=0.3",
            facecolor="silver",
            edgecolor="black",
            linewidth=3,
        )
        patches_list.append(counter)

        # Bếp nấu (3 vòng tròn)
        for i in range(3):
            stove = Circle(
                (x - 2 + i * 2, y + height / 2),
                0.5,
                facecolor="darkgray",
                edgecolor="black",
                linewidth=2,
            )
            patches_list.append(stove)

        # Kệ phía sau
        shelf = patches.Rectangle(
            (x - width / 2 + 0.5, y + height + 0.2),
            width - 1,
            0.8,
            facecolor="burlywood",
            edgecolor="black",
            linewidth=1,
        )
        patches_list.append(shelf)

        return patches_list

    @staticmethod
    def create_walls(bounds=(-15, 15, -15, 15), door_positions=None):
        """Tạo tường bao quanh"""
        patches_list = []
        xmin, xmax, ymin, ymax = bounds
        wall_thickness = 0.5

        # Tường trái
        patches_list.append(
            patches.Rectangle(
                (xmin, ymin),
                wall_thickness,
                ymax - ymin,
                facecolor="gray",
                edgecolor="black",
            )
        )

        # Tường phải
        patches_list.append(
            patches.Rectangle(
                (xmax - wall_thickness, ymin),
                wall_thickness,
                ymax - ymin,
                facecolor="gray",
                edgecolor="black",
            )
        )

        # Tường dưới
        patches_list.append(
            patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                wall_thickness,
                facecolor="gray",
                edgecolor="black",
            )
        )

        return patches_list

    @staticmethod
    def create_floor_pattern(ax, bounds=(-15, 15, -15, 15)):
        """Tạo hoa văn sàn nhà"""
        xmin, xmax, ymin, ymax = bounds

        # Sàn gỗ pattern
        for i in range(int(xmin), int(xmax), 2):
            for j in range(int(ymin), int(ymax), 2):
                color = "#D2B48C" if (i + j) % 4 == 0 else "#C4A77D"
                rect = patches.Rectangle(
                    (i, j),
                    2,
                    2,
                    facecolor=color,
                    edgecolor="#8B7355",
                    linewidth=0.5,
                    alpha=0.3,
                    zorder=0,
                )
                ax.add_patch(rect)


class DynamicObstacleManager:
    """Quản lý các vật cản động (khách hàng)"""

    def __init__(self, num_customers=3):
        self.customers = []
        self.avoid_areas = []  # Vùng cần tránh (bàn, bếp, ...)

        # Tạo khách hàng
        colors = ["orange", "coral", "gold", "tomato", "salmon"]
        for i in range(num_customers):
            # Vị trí khởi tạo ngẫu nhiên
            x = np.random.uniform(-8, 8)
            y = np.random.uniform(-10, 8)
            color = colors[i % len(colors)]
            customer = Customer(x, y, i, color)
            customer.wait_time = np.random.uniform(0, 3)
            self.customers.append(customer)

    def set_avoid_areas(self, areas):
        """Đặt các vùng cần tránh"""
        self.avoid_areas = areas
        for customer in self.customers:
            customer.set_random_target(self.avoid_areas)

    def update(self, dt, static_obstacles=None, robot=None):
        """Cập nhật tất cả khách hàng"""
        for customer in self.customers:
            customer.update(
                dt, static_obstacles, self.customers, self.avoid_areas, robot=robot
            )
            customer.update_graphics()

    def add_to_axes(self, ax):
        """Thêm đồ họa khách hàng vào axes"""
        for customer in self.customers:
            for patch in customer.get_patch_list():
                ax.add_patch(patch)

    def get_positions(self):
        """Lấy vị trí tất cả khách hàng"""
        return [(c.x, c.y, c.radius) for c in self.customers]

    def check_collision(self, x, y, radius=0.5):
        """Kiểm tra va chạm với khách hàng"""
        for customer in self.customers:
            dist = np.hypot(x - customer.x, y - customer.y)
            if dist < radius + customer.radius:
                return True
        return False
