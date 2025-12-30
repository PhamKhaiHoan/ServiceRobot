import tkinter as tk
import matplotlib.patches as patches
import Robot as rob
import Obstacle as obs_set
from GUI import RobotSimulatorGUI

# --- 1. CẤU HÌNH ĐỊA ĐIỂM (LOCATIONS) ---
# Tọa độ điểm đến AN TOÀN (Robot sẽ đi đến đây)
LOCATIONS = {
    "Kitchen": (0, 10),  # Bếp (Trên cùng)
    "Table 1": (-7, 6),  # Bàn 1 (Trái trên)
    "Table 2": (7, 6),  # Bàn 2 (Phải trên)
    "Table 3": (-7, 0),  # Bàn 3 (Trái giữa)
    "Table 4": (7, 0),  # Bàn 4 (Phải giữa)
    "Table 5": (-7, -6),  # Bàn 5 (Trái dưới)
    "Table 6": (7, -6),  # Bàn 6 (Phải dưới)
}

# --- 2. TẠO VẬT CẢN (VISUALS) ---
obstacles = obs_set.Obstacle()

# Bếp
obstacles.add(patches.Rectangle((-4, 12), 8, 2, color="black"))

# 6 Bàn (Hình vuông 3x3)
obstacles.add(patches.Rectangle((-13, 5), 3, 3, color="brown"))  # Bàn 1
obstacles.add(patches.Rectangle((10, 5), 3, 3, color="brown"))  # Bàn 2
obstacles.add(patches.Rectangle((-13, -1), 3, 3, color="brown"))  # Bàn 3
obstacles.add(patches.Rectangle((10, -1), 3, 3, color="brown"))  # Bàn 4
obstacles.add(patches.Rectangle((-13, -7), 3, 3, color="brown"))  # Bàn 5
obstacles.add(patches.Rectangle((10, -7), 3, 3, color="brown"))  # Bàn 6

# --- 3. KHỞI TẠO ROBOT ---
dt = 0.1
robot = rob.Differential_Robot(
    L=1.0, robot_length=1.5, robot_width=1.0, wheel_length=0.5, wheel_width=0.2
)
robot.x, robot.y = LOCATIONS["Kitchen"]  # Xuất phát từ Bếp
robot.theta = -1.57  # Hướng mặt xuống dưới

# --- 4. CHẠY ---
root = tk.Tk()
root.title("Restaurant Service Robot")

# Truyền locations vào GUI
app = RobotSimulatorGUI(root, robot, obstacles, locations=LOCATIONS, dt=dt)

root.mainloop()
