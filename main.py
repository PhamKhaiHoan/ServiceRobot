import tkinter as tk
import matplotlib.patches as patches
import Robot as rob
import Obstacle as obs_set
from GUI import RobotSimulatorGUI

# --- 1. THÔNG SỐ CƠ BẢN ---
dt = 0.1
L = 1.0

# Khởi tạo robot
robot = rob.Differential_Robot(
    L, robot_length=2.0, robot_width=1.0, wheel_length=0.5, wheel_width=0.5
)

# --- 2. TẠO BẢN ĐỒ ĐỒNG BỘ (BÀN VUÔNG 2x2) ---
obstacles = obs_set.Obstacle()

# Quy ước: patches.Rectangle((x_góc_trái_dưới, y_góc_trái_dưới), chiều_rộng, chiều_cao)
# Tạo 4 bàn giống hệt nhau (2m x 2m)
obstacles.add(patches.Rectangle((-8, -8), 2, 2, color="brown"))  # Bàn 1 (Góc dưới trái)
obstacles.add(patches.Rectangle((6, -8), 2, 2, color="brown"))  # Bàn 2 (Góc dưới phải)
obstacles.add(patches.Rectangle((6, 6), 2, 2, color="brown"))  # Bàn 3 (Góc trên phải)
obstacles.add(patches.Rectangle((-8, 6), 2, 2, color="brown"))  # Bàn 4 (Góc trên trái)

# Thêm một Bếp (Vùng cấm) ở giữa trên cùng
obstacles.add(patches.Rectangle((-1, 12), 2, 1, color="gray"))

# --- 3. KHỞI TẠO GUI ---
root = tk.Tk()
root.title("Autonomous Serving Robot - Uniform Map")
app = RobotSimulatorGUI(root, robot, obstacles, dt=dt)

# --- 4. KỊCH BẢN CHẠY (CẬP NHẬT TỌA ĐỘ AN TOÀN) ---
# Tọa độ điểm đến = Tọa độ bàn +/- khoảng cách an toàn (tránh va chạm)

# Bếp (về vị trí (0,0) là an toàn nhất vì bếp ở tận y=12)
KITCHEN = (0, 0)

# Bàn 1 ở (-8, -8) -> Điểm đến (-5, -7) (Đứng bên phải bàn)
TABLE_1 = (-5, -7)

# Bàn 2 ở (6, -8) -> Điểm đến (5, -5) (Đứng phía trên bàn)
TABLE_2 = (5, -5)

# Bàn 3 ở (6, 6) -> Điểm đến (5, 5) (Đứng phía dưới góc trái bàn)
TABLE_3 = (5, 5)

# Nạp nhiệm vụ
app.destination_queue = [TABLE_1, KITCHEN, TABLE_2, TABLE_3, KITCHEN]
app.auto_mode = True

print("Hệ thống đã khởi động với bản đồ đồng bộ.")
print("Lịch trình: Bàn 1 -> Bếp -> Bàn 2 -> Bàn 3 -> Bếp")

root.mainloop()
