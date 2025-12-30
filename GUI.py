import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path
import time

# Import tất cả các planner
from DWA_planner import DWA_Planner
from bug_planner import BugPlanner
from astar_planner import AStarPlanner
from dijkstra_planner import DijkstraPlanner


class RobotSimulatorGUI:
    def __init__(self, root, robot, obstacles, dt=0.1):
        self.root = root
        self.robot = robot
        self.obstacles = obstacles
        self.dt = dt

        # --- QUẢN LÝ HÀNG ĐỢI (ROLE 1) ---
        self.destination_queue = []
        self.is_waiting = False
        self.wait_start_time = 0
        self.wait_duration = 2.0

        # --- QUẢN LÝ THUẬT TOÁN (ROLE 2) ---
        self.current_algo = "DWA"  # Mặc định
        # Khởi tạo các planner
        self.dwa = DWA_Planner(robot, obstacles, dt, predict_time=2.0)
        self.bug = BugPlanner(robot, obstacles, step_size=0.5)
        self.astar = AStarPlanner(obstacles, resolution=0.5, robot_radius=1.0)
        self.dijkstra = DijkstraPlanner(obstacles, resolution=0.5, robot_radius=1.0)

        # Biến phục vụ việc bám đường (Path Following)
        self.global_path = []  # Chứa danh sách các điểm [ (x1,y1), (x2,y2), ... ]
        self.current_waypoint_index = 0

        # Các cờ trạng thái
        self.auto_mode = False
        self.destination = None
        self.destination_marker = None

        # Đồ họa
        self.x_data, self.y_data = [], []
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-12, 12)  # Zoom map cho vừa mắt
        self.ax.set_ylim(-12, 16)
        self.ax.set_aspect("equal")
        self.ax.grid()

        for obs in self.obstacles.obs:
            self.ax.add_patch(obs)

        (self.path_line,) = self.ax.plot([], [], "b-", label="Path", linewidth=1)
        (self.global_path_line,) = self.ax.plot(
            [], [], "g--", label="Global Plan", alpha=0.5
        )  # Vẽ đường dự kiến
        (self.robot_dot,) = self.ax.plot([], [], "ro", label="Robot")

        self.ax.add_patch(self.robot.robot_body)
        self.ax.add_patch(self.robot.left_wheel)
        self.ax.add_patch(self.robot.right_wheel)
        self.ax.add_patch(self.robot.heading_arrow)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(
            row=0, column=0, rowspan=15, sticky="nsew", padx=10, pady=10
        )
        self.canvas.draw()

        self._create_controls()
        self._create_status_display()

        self.root.bind("<KeyPress>", self.key_press)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.ani = FuncAnimation(
            self.fig, self.update, interval=int(self.dt * 1000), blit=False
        )

    def _create_controls(self):
        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=0, column=1, sticky="n", padx=10)

        # Nút điều khiển xe
        ttk.Label(btn_frame, text="MANUAL CONTROL", font=("Arial", 10, "bold")).grid(
            row=0, column=0, columnspan=3, pady=5
        )
        ttk.Button(btn_frame, text="↑", command=self.move_forward).grid(row=1, column=1)
        ttk.Button(btn_frame, text="←", command=self.turn_left).grid(row=2, column=0)
        ttk.Button(btn_frame, text="STOP", command=self.stop).grid(row=2, column=1)
        ttk.Button(btn_frame, text="→", command=self.turn_right).grid(row=2, column=2)
        ttk.Button(btn_frame, text="↓", command=self.move_backward).grid(
            row=3, column=1
        )

        # Nút chọn thuật toán
        ttk.Label(btn_frame, text="ALGORITHM", font=("Arial", 10, "bold")).grid(
            row=4, column=0, columnspan=3, pady=(15, 5)
        )
        self.algo_var = tk.StringVar(value="DWA")
        algos = ["DWA", "BUG", "A_STAR", "DIJKSTRA"]
        for i, mode in enumerate(algos):
            rb = ttk.Radiobutton(
                btn_frame,
                text=mode,
                variable=self.algo_var,
                value=mode,
                command=self.change_algo,
            )
            rb.grid(row=5 + i, column=0, columnspan=3, sticky="w")

        ttk.Button(btn_frame, text="RESET", command=self.reset).grid(
            row=10, column=0, columnspan=3, pady=15
        )

    def _create_status_display(self):
        status_frame = ttk.LabelFrame(self.root, text="Info", padding=10)
        status_frame.grid(row=1, column=1, sticky="nw", padx=10)
        self.status_label = ttk.Label(
            status_frame, text="Ready", foreground="blue", font=("Arial", 11)
        )
        self.status_label.pack()

    def change_algo(self):
        self.current_algo = self.algo_var.get()
        print(f"Switched to: {self.current_algo}")
        self.global_path = []  # Reset đường cũ khi đổi thuật toán

    def update(self, frame):
        # 1. LOGIC TỰ ĐỘNG (AUTO MODE)
        # Lấy nhiệm vụ từ hàng đợi nếu đang rảnh
        if self.destination is None and self.destination_queue:
            self.destination = self.destination_queue.pop(0)
            self.global_path = []  # Reset đường global
            self.current_waypoint_index = 0

            # Nếu dùng A* hoặc Dijkstra, tính toán đường đi NGAY LÚC NÀY
            if self.current_algo in ["A_STAR", "DIJKSTRA"]:
                print(f"Planning path using {self.current_algo}...")
                start = (self.robot.x, self.robot.y)
                if self.current_algo == "A_STAR":
                    self.global_path = self.astar.plan(start, self.destination)
                else:
                    self.global_path = self.dijkstra.plan(start, self.destination)

                if not self.global_path:
                    print("Failed to find path!")
                    self.status_label.config(text="No Path Found!", foreground="red")
                    self.destination = None  # Bỏ qua điểm này
                else:
                    print(f"Path found with {len(self.global_path)} steps.")
                    # Vẽ đường dự kiến lên map
                    path_x = [p[0] for p in self.global_path]
                    path_y = [p[1] for p in self.global_path]
                    self.global_path_line.set_data(path_x, path_y)

            # Vẽ marker đích
            if self.destination_marker:
                self.destination_marker.remove()
            (self.destination_marker,) = self.ax.plot(
                self.destination[0], self.destination[1], "r*", markersize=15
            )

        # Xử lý di chuyển
        if self.auto_mode and self.destination is not None:
            dist_to_goal = np.hypot(
                self.destination[0] - self.robot.x, self.destination[1] - self.robot.y
            )

            # Kiểm tra đến đích
            if dist_to_goal < 0.3:
                self.stop()
                if not self.is_waiting:
                    self.is_waiting = True
                    self.wait_start_time = time.time()
                    self.status_label.config(text="Serving...", foreground="green")
                elif time.time() - self.wait_start_time > self.wait_duration:
                    self.is_waiting = False
                    self.destination = None
                    self.global_path = []
                    self.global_path_line.set_data([], [])  # Xóa đường vẽ
                    if self.destination_marker:
                        self.destination_marker.remove()
                        self.destination_marker = None
                    if not self.destination_queue:
                        self.status_label.config(text="All Done!", foreground="blue")

            # Chưa đến đích -> Tính toán di chuyển
            else:
                v_l, v_r = 0, 0

                # --- TRƯỜNG HỢP 1: DWA hoặc BUG (Chạy trực tiếp) ---
                if self.current_algo == "DWA":
                    v_l, v_r = self.dwa.plan(self.destination)
                elif self.current_algo == "BUG":
                    v_l, v_r = self.bug.plan(
                        (self.robot.x, self.robot.y), self.destination
                    )

                # --- TRƯỜNG HỢP 2: A* hoặc DIJKSTRA (Bám theo đường) ---
                elif self.current_algo in ["A_STAR", "DIJKSTRA"] and self.global_path:
                    # Tìm điểm waypoint tiếp theo trong danh sách global_path
                    # Logic: Nếu robot đến gần waypoint hiện tại (< 0.5m), chuyển sang waypoint kế tiếp
                    if self.current_waypoint_index < len(self.global_path):
                        local_target = self.global_path[self.current_waypoint_index]
                        dist_to_waypoint = np.hypot(
                            local_target[0] - self.robot.x,
                            local_target[1] - self.robot.y,
                        )

                        if dist_to_waypoint < 0.5:
                            self.current_waypoint_index += 1  # Sang điểm tiếp theo

                        # Dùng DWA để lái đến điểm waypoint đó (Local Planner follow Global Path)
                        # Nếu hết đường thì lái thẳng đến đích gốc
                        if self.current_waypoint_index < len(self.global_path):
                            target_now = self.global_path[self.current_waypoint_index]
                        else:
                            target_now = self.destination

                        v_l, v_r = self.dwa.plan(target_now)

                # Cập nhật vận tốc
                self.robot.v_l = v_l
                self.robot.v_r = v_r

                # Hiển thị
                self.status_label.config(
                    text=f"Mode: {self.current_algo}\nDist: {dist_to_goal:.2f}m",
                    foreground="black",
                )

        # 2. CẬP NHẬT VẬT LÝ
        if self.robot.v_l != 0 or self.robot.v_r != 0:
            self.robot.update_pose(self.dt)
            self.x_data.append(self.robot.x)
            self.y_data.append(self.robot.y)

        # 3. CẬP NHẬT ĐỒ HỌA
        self.path_line.set_data(self.x_data, self.y_data)
        self.robot_dot.set_data([self.robot.x], [self.robot.y])
        self.robot.update_graphics(self.ax)
        self.canvas.draw_idle()
        return (self.path_line,)

    # --- CÁC HÀM HỖ TRỢ ---
    def check_collision(self, x, y, theta):
        # (Giữ nguyên logic va chạm cũ của bạn nếu cần dùng BugPlanner để check)
        return False

    def move_forward(self):
        self.robot.v_l = 1.5
        self.robot.v_r = 1.5
        self.auto_mode = False

    def move_backward(self):
        self.robot.v_l = -1.5
        self.robot.v_r = -1.5
        self.auto_mode = False

    def turn_left(self):
        self.robot.v_l = -0.5
        self.robot.v_r = 0.5
        self.auto_mode = False

    def turn_right(self):
        self.robot.v_l = 0.5
        self.robot.v_r = -0.5
        self.auto_mode = False

    def stop(self):
        self.robot.v_l = 0
        self.robot.v_r = 0

    def reset(self):
        self.auto_mode = False
        self.destination_queue = []
        self.destination = None
        self.global_path = []
        self.global_path_line.set_data([], [])
        if self.destination_marker:
            self.destination_marker.remove()
        self.robot.reset()
        self.x_data.clear()
        self.y_data.clear()
        self.path_line.set_data([], [])
        self.robot_dot.set_data([], [])
        self.canvas.draw_idle()

    def key_press(self, event):
        if event.keysym == "Up":
            self.move_forward()
        elif event.keysym == "Down":
            self.move_backward()
        elif event.keysym == "Left":
            self.turn_left()
        elif event.keysym == "Right":
            self.turn_right()
        elif event.keysym == "space":
            self.stop()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        target = (event.xdata, event.ydata)
        self.destination_queue.append(target)
        self.auto_mode = True
        self.ax.plot(target[0], target[1], "bx", markersize=8)  
        self.canvas.draw_idle()
        print(f"Added task: {target}")
