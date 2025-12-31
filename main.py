import tkinter as tk
import matplotlib.patches as patches
import Robot as rob
import Obstacle as obs_set
from Obstacle import RestaurantMap, DynamicObstacleManager
from GUI import RobotSimulatorGUI

# --- 1. CẤU HÌNH ĐỊA ĐIỂM (LOCATIONS) ---
# Tọa độ điểm đến AN TOÀN - Robot sẽ đứng CẠNH bàn (không phải giữa bàn)
LOCATIONS = {
    "Kitchen": (0, 10),  # Bếp (phía trước quầy)
    "Table 1": (-5, 5),  # Cạnh bàn 1 (phía phải của bàn)
    "Table 2": (5, 5),  # Cạnh bàn 2 (phía trái của bàn)
    "Table 3": (-5, -1),  # Cạnh bàn 3 (phía phải của bàn)
    "Table 4": (5, -1),  # Cạnh bàn 4 (phía trái của bàn)
    "Table 5": (-5, -7),  # Cạnh bàn 5 (phía phải của bàn)
    "Table 6": (5, -7),  # Cạnh bàn 6 (phía trái của bàn)
}

# Vùng bàn để tránh (x, y, width, height)
TABLE_AREAS = [
    (-10, 4, 4, 4),  # Bàn 1
    (6, 4, 4, 4),  # Bàn 2
    (-10, -3, 4, 4),  # Bàn 3
    (6, -3, 4, 4),  # Bàn 4
    (-10, -9, 4, 4),  # Bàn 5
    (6, -9, 4, 4),  # Bàn 6
    (-5, 11, 10, 4),  # Bếp
]

# --- 2. TẠO VẬT CẢN (VISUALS) ---
obstacles = obs_set.Obstacle()

# Tạo bếp đẹp
for patch in RestaurantMap.create_kitchen(0, 11.5, width=10, height=2.5):
    obstacles.add(patch)

# Tạo 6 bàn (không có ghế)
TABLE_POSITIONS = [
    (-8, 5, "square", "Table 1"),
    (8, 5, "round", "Table 2"),
    (-8, -1, "square", "Table 3"),
    (8, -1, "round", "Table 4"),
    (-8, -7, "square", "Table 5"),
    (8, -7, "round", "Table 6"),
]

for tx, ty, ttype, tname in TABLE_POSITIONS:
    for patch in RestaurantMap.create_table(tx, ty, ttype, tname):
        obstacles.add(patch)

# Tạo tường
for patch in RestaurantMap.create_walls(bounds=(-14, 14, -12, 15)):
    obstacles.add(patch)

# --- 3. TẠO KHÁCH HÀNG DI CHUYỂN ---
customer_manager = DynamicObstacleManager(num_customers=4)
customer_manager.set_avoid_areas(TABLE_AREAS)

# --- 4. KHỞI TẠO ROBOT ---
dt = 0.1
robot = rob.Differential_Robot(
    L=1.0, robot_length=1.5, robot_width=1.0, wheel_length=0.5, wheel_width=0.2
)
robot.x, robot.y = LOCATIONS["Kitchen"]  # Xuất phát từ Bếp
robot.theta = -1.57  # Hướng mặt xuống dưới

# --- 5. CHẠY ---
root = tk.Tk()
root.title("Restaurant Service Robot")

# Truyền locations và customer_manager vào GUI
app = RobotSimulatorGUI(
    root,
    robot,
    obstacles,
    locations=LOCATIONS,
    dt=dt,
    customer_manager=customer_manager,
)

root.mainloop()
