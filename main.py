import tkinter as tk
import matplotlib.patches as patches
import Robot as rob
import Obstacle as obs_set
from GUI import RobotSimulatorGUI

# from DWA_planner import DWA_Planner # GUI đã tự init planner rồi

# Thông số mô phỏng
dt = 0.1
L = 1.0

# Khởi tạo robot
robot = rob.Differential_Robot(
    L, robot_length=2.0, robot_width=1.0, wheel_length=0.5, wheel_width=0.5
)

# Khởi tạo chướng ngại vật (Mô phỏng bàn ghế)
obstacles = obs_set.Obstacle()
obstacles.add(patches.Rectangle((-6, -5), 3.2, 3.2, color="gray"))  # Bàn 1
obstacles.add(patches.Circle((10, 11), 2, color="gray"))  # Bàn 2
obstacles.add(patches.Rectangle((5, -10), 6, 2.3, color="dimgray"))  # Bàn 3
obstacles.add(patches.Circle((-12, -12), 2.5, color="gray"))  # Bàn 4

# Khởi tạo giao diện
root = tk.Tk()
root.title("Autonomous Serving Robot - Role 1 Logic")
app = RobotSimulatorGUI(root, robot, obstacles, dt=dt)

# --- PHẦN NGƯỜI 1: SETUP KỊCH BẢN ---
# Định nghĩa tọa độ
KITCHEN = (0, 0)
TABLE_1 = (-6, -3)  # Gần bàn chữ nhật trái
TABLE_2 = (10, 9)  # Gần bàn tròn phải trên
TABLE_3 = (5, -8)  # Gần bàn chữ nhật phải dưới

# Nạp sẵn nhiệm vụ
app.destination_queue = [TABLE_1, KITCHEN, TABLE_2, TABLE_3, KITCHEN]
app.auto_mode = True

print("System Started. Task Queue: Table 1 -> Kitchen -> Table 2 -> Table 3 -> Kitchen")
# ------------------------------------

root.mainloop()
