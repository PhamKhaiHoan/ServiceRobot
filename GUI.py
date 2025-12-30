import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import time

# Import các thuật toán
from DWA_planner import DWA_Planner
from bug_planner import BugPlanner
from astar_planner import AStarPlanner
from dijkstra_planner import DijkstraPlanner


class RobotSimulatorGUI:
    def __init__(self, root, robot, obstacles, locations, dt=0.1):
        self.root = root
        self.robot = robot
        self.obstacles = obstacles
        self.locations = locations  # Dict chứa tọa độ bàn/bếp
        self.dt = dt

        # --- TRẠNG THÁI HỆ THỐNG ---
        self.destination_queue = []  # Hàng đợi tọa độ [(x,y), ...]
        self.task_names = []  # Tên nhiệm vụ để hiển thị ["Table 1", "Kitchen"...]
        self.is_executing = False  # Cờ cho phép chạy (True = đang chạy)
        self.is_waiting = False  # Cờ đang phục vụ (dừng 2s)
        self.wait_start_time = 0
        self.wait_duration = 2.0
        self.destination = None  # Điểm đến hiện tại

        # --- PATH PLANNING ---
        self.current_algo = "DWA"  # Mặc định

        # --- KHỞI TẠO PLANNERS (SỬA LỖI TẠI ĐÂY) ---
        self.dwa = DWA_Planner(robot, obstacles, dt, predict_time=2.0)

        # [FIX] BugPlanner không nhận dt, chỉ nhận step_size
        self.bug = BugPlanner(robot, obstacles, step_size=0.5)

        self.astar = AStarPlanner(obstacles, resolution=0.5, robot_radius=1.0)
        self.dijkstra = DijkstraPlanner(obstacles, resolution=0.5, robot_radius=1.0)

        self.global_path = []
        self.current_wp_index = 0

        # --- ĐỒ HỌA (MATPLOTLIB) ---
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.5)

        # Vẽ vật cản
        for obs in self.obstacles.obs:
            self.ax.add_patch(obs)

        # Vẽ tên địa điểm lên bản đồ
        for name, (x, y) in self.locations.items():
            self.ax.text(
                x, y, name, color="blue", fontsize=8, fontweight="bold", ha="center"
            )
            self.ax.plot(x, y, "b+", markersize=5)

        # Robot & Path
        self.robot_body = self.ax.add_patch(self.robot.robot_body)
        self.left_wheel = self.ax.add_patch(self.robot.left_wheel)
        self.right_wheel = self.ax.add_patch(self.robot.right_wheel)
        self.heading_arrow = self.ax.add_patch(self.robot.heading_arrow)
        (self.path_line,) = self.ax.plot([], [], "r-", linewidth=1, alpha=0.5)
        (self.global_path_line,) = self.ax.plot([], [], "g--", linewidth=1.5, alpha=0.7)
        (self.destination_marker,) = self.ax.plot([], [], "r*", markersize=12)

        self.x_data, self.y_data = [], []

        # --- CANVAS ---
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(
            row=0, column=0, rowspan=20, sticky="nsew", padx=5, pady=5
        )

        # --- CONTROL PANEL ---
        self._create_controls()

        # Animation Loop
        self.ani = FuncAnimation(
            self.fig, self.update, interval=int(dt * 1000), blit=False
        )

    def _create_controls(self):
        panel = ttk.Frame(self.root)
        panel.grid(row=0, column=1, sticky="n", padx=10, pady=10)

        # 1. Chọn Logic
        ttk.Label(panel, text="LOGIC MODE", font=("Arial", 10, "bold")).pack(
            pady=(0, 5)
        )

        self.algo_var = tk.StringVar(value="DWA")
        frame_algo = ttk.Frame(panel)
        frame_algo.pack()

        algos = ["DWA", "BUG", "A_STAR", "DIJKSTRA"]
        for alg in algos:
            rb = ttk.Radiobutton(
                frame_algo,
                text=alg,
                variable=self.algo_var,
                value=alg,
                command=self.change_algo,
            )
            rb.pack(anchor="w")

        # 2. Chọn Bàn (Queue)
        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(panel, text="SELECT TABLES", font=("Arial", 10, "bold")).pack()

        frame_tables = ttk.Frame(panel)
        frame_tables.pack(pady=5)

        # Tạo nút bấm động dựa trên danh sách locations
        i = 0
        sorted_locs = sorted([k for k in self.locations.keys() if k != "Kitchen"])
        for name in sorted_locs:
            btn = ttk.Button(
                frame_tables,
                text=f"Add {name}",
                command=lambda n=name: self.add_to_queue(n),
            )
            btn.grid(row=i // 2, column=i % 2, padx=2, pady=2)
            i += 1

        # 3. Điều khiển chạy
        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=10)

        self.lbl_queue = ttk.Label(
            panel, text="Queue: Empty", foreground="gray", wraplength=150
        )
        self.lbl_queue.pack(pady=5)

        btn_start = ttk.Button(
            panel, text="▶ START SERVICE", command=self.start_service
        )
        btn_start.pack(fill="x", pady=2)

        btn_clear = ttk.Button(panel, text="✖ CLEAR QUEUE", command=self.clear_queue)
        btn_clear.pack(fill="x", pady=2)

        self.lbl_status = ttk.Label(
            panel, text="Status: Idle", foreground="blue", font=("Arial", 11, "bold")
        )
        self.lbl_status.pack(pady=10)

    # --- LOGIC CONTROL ---
    def change_algo(self):
        self.current_algo = self.algo_var.get()
        self.global_path = []
        print(f"Algorithm changed to: {self.current_algo}")

    def add_to_queue(self, name):
        if name in self.locations:
            self.destination_queue.append(self.locations[name])
            self.task_names.append(name)
            self.update_queue_label()

    def update_queue_label(self):
        text = " -> ".join(self.task_names)
        self.lbl_queue.config(text=f"Queue: {text}")

    def clear_queue(self):
        self.destination_queue = []
        self.task_names = []
        self.is_executing = False
        self.destination = None
        self.global_path = []
        self.update_queue_label()
        self.lbl_status.config(text="Status: Stopped & Cleared")

    def start_service(self):
        if not self.destination_queue:
            self.lbl_status.config(text="Status: Queue Empty!")
            return

        # Tự động thêm 'Kitchen' vào cuối hàng đợi để xe quay về
        if not self.task_names or self.task_names[-1] != "Kitchen":
            self.destination_queue.append(self.locations["Kitchen"])
            self.task_names.append("Kitchen")
            self.update_queue_label()

        self.is_executing = True
        self.lbl_status.config(text="Status: Executing...")

    # --- UPDATE LOOP ---
    def update(self, frame):
        if self.is_executing:
            # 1. Lấy nhiệm vụ mới
            if self.destination is None and self.destination_queue:
                self.destination = self.destination_queue.pop(0)
                current_task = self.task_names.pop(0)
                self.update_queue_label()
                self.lbl_status.config(text=f"Going to: {current_task}")

                # Reset path cũ
                self.global_path = []
                self.current_wp_index = 0

                # Plan đường (A*/Dijkstra)
                if self.current_algo in ["A_STAR", "DIJKSTRA"]:
                    start_pos = (self.robot.x, self.robot.y)
                    if self.current_algo == "A_STAR":
                        self.global_path = self.astar.plan(start_pos, self.destination)
                    else:
                        self.global_path = self.dijkstra.plan(
                            start_pos, self.destination
                        )

                    if self.global_path:
                        gx = [p[0] for p in self.global_path]
                        gy = [p[1] for p in self.global_path]
                        self.global_path_line.set_data(gx, gy)
                    else:
                        print("Không tìm thấy đường!")
                        self.destination = None

            # 2. Di chuyển
            if self.destination is not None:
                self.destination_marker.set_data(
                    [self.destination[0]], [self.destination[1]]
                )
                dist = np.hypot(
                    self.destination[0] - self.robot.x,
                    self.destination[1] - self.robot.y,
                )

                # Đến đích
                if dist < 0.3:
                    self.stop_robot()
                    if not self.is_waiting:
                        self.is_waiting = True
                        self.wait_start_time = time.time()
                        self.lbl_status.config(text="Arrived! Serving...")
                    elif time.time() - self.wait_start_time > self.wait_duration:
                        self.is_waiting = False
                        self.destination = None
                        if not self.destination_queue:
                            self.is_executing = False
                            self.lbl_status.config(text="Status: All Done! At Kitchen.")

                # Đang đi
                elif not self.is_waiting:
                    v_l, v_r = 0, 0

                    if self.current_algo == "DWA":
                        v_l, v_r = self.dwa.plan(self.destination)
                    elif self.current_algo == "BUG":
                        v_l, v_r = self.bug.plan(
                            (self.robot.x, self.robot.y), self.destination
                        )
                    elif (
                        self.current_algo in ["A_STAR", "DIJKSTRA"] and self.global_path
                    ):
                        # Follow path logic
                        if self.current_wp_index < len(self.global_path):
                            target_wp = self.global_path[self.current_wp_index]
                            dist_wp = np.hypot(
                                target_wp[0] - self.robot.x, target_wp[1] - self.robot.y
                            )
                            if dist_wp < 0.5:
                                self.current_wp_index += 1

                            drive_target = self.global_path[
                                min(self.current_wp_index, len(self.global_path) - 1)
                            ]
                            v_l, v_r = self.dwa.plan(drive_target)

                    self.robot.v_l = v_l
                    self.robot.v_r = v_r

            self.robot.update_pose(self.dt)
            self.x_data.append(self.robot.x)
            self.y_data.append(self.robot.y)

        # Update Graphics
        self.path_line.set_data(self.x_data, self.y_data)
        self.robot.update_graphics(self.ax)
        self.canvas.draw_idle()

    def stop_robot(self):
        self.robot.v_l = 0
        self.robot.v_r = 0
