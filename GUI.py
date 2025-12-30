import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path
import time

# Import planner (Chỉ giữ lại DWA cho Người 1 test logic)
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

        # --- PHẦN NGƯỜI 1: QUẢN LÝ HÀNG ĐỢI ---
        self.destination_queue = []  # Hàng đợi các điểm đến
        self.is_waiting = False  # Trạng thái đang phục vụ/chờ
        self.wait_start_time = 0
        self.wait_duration = 2.0  # Thời gian dừng tại mỗi điểm (giây)
        # --------------------------------------

        # Khởi tạo DWA planner mặc định
        self.planner = DWA_Planner(
            self.robot, self.obstacles, dt=self.dt, predict_time=2.0
        )

        # Các cờ trạng thái
        self.auto_mode = False
        self.destination = None
        self.destination_marker = None

        # Dữ liệu vẽ đường đi
        self.x_data, self.y_data = [], []

        # Setup đồ họa matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect("equal")
        self.ax.grid()

        # Vẽ chướng ngại vật
        for obs in self.obstacles.obs:
            self.ax.add_patch(obs)

        # Vẽ robot và đường đi
        (self.path_line,) = self.ax.plot([], [], "b-", label="Path")
        (self.robot_dot,) = self.ax.plot([], [], "ro", label="Robot")

        self.ax.add_patch(self.robot.robot_body)
        self.ax.add_patch(self.robot.left_wheel)
        self.ax.add_patch(self.robot.right_wheel)
        self.ax.add_patch(self.robot.heading_arrow)
        self.ax.add_patch(self.robot.left_arrow)
        self.ax.add_patch(self.robot.right_arrow)

        # Setup Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(
            row=0, column=0, rowspan=15, sticky="nsew", padx=10, pady=10
        )
        self.canvas.draw()

        # Controls & Status
        self._create_controls()
        self._create_status_display()

        # Events
        self.root.bind("<KeyPress>", self.key_press)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        # Animation Loop
        self.ani = FuncAnimation(
            self.fig, self.update, interval=int(self.dt * 1000), blit=False
        )

    def _create_controls(self):
        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=0, column=1, sticky="n", padx=10)

        ttk.Button(btn_frame, text="↑ Forward", command=self.move_forward).grid(
            row=0, column=1
        )
        ttk.Button(btn_frame, text="Left", command=self.turn_left).grid(row=1, column=0)
        ttk.Button(btn_frame, text="Stop", command=self.stop).grid(row=1, column=1)
        ttk.Button(btn_frame, text="Right", command=self.turn_right).grid(
            row=1, column=2
        )
        ttk.Button(btn_frame, text="↓ Backward", command=self.move_backward).grid(
            row=2, column=1
        )
        ttk.Button(btn_frame, text="Reset", command=self.reset).grid(row=3, column=1)

    def _create_status_display(self):
        status_frame = ttk.LabelFrame(self.root, text="Status", padding=10)
        status_frame.grid(row=1, column=1, sticky="nw", padx=10, pady=10)

        self.collision_label = ttk.Label(
            status_frame, text="Ready", foreground="red", font=("Arial", 12, "bold")
        )
        self.collision_label.grid(row=4, column=0, sticky="w", pady=10)

    def check_collision(self, x, y, theta):
        robot_corners = self.robot.get_robot_corners(x, y, theta)
        robot_path = Path(robot_corners)
        for obs in self.obstacles.obs:
            if isinstance(obs, patches.Rectangle):
                bbox = obs.get_bbox()
                rect_corners = np.array(
                    [
                        [bbox.xmin, bbox.ymin],
                        [bbox.xmin, bbox.ymax],
                        [bbox.xmax, bbox.ymax],
                        [bbox.xmax, bbox.ymin],
                        [bbox.xmin, bbox.ymin],
                    ]
                )
                obs_path = Path(rect_corners)
                if robot_path.intersects_path(obs_path, filled=True):
                    return True
            elif isinstance(obs, patches.Circle):
                center = np.array(obs.center)
                radius = obs.radius
                if np.any(
                    [
                        np.linalg.norm(corner - center) <= radius
                        for corner in robot_corners
                    ]
                ):
                    return True
                if robot_path.contains_point(center):
                    return True
        return False

    def update(self, frame):
        # --- 1. LOGIC TỰ ĐỘNG (AUTO MODE) ---
        # Nếu đang auto, tính toán vận tốc dựa trên DWA/Planner
        if self.destination is None and self.destination_queue:
            self.destination = self.destination_queue.pop(0)
            if self.destination_marker:
                self.destination_marker.remove()
            (self.destination_marker,) = self.ax.plot(
                self.destination[0], self.destination[1], "r*", markersize=15
            )

        if self.auto_mode and self.destination is not None:
            dx = self.destination[0] - self.robot.x
            dy = self.destination[1] - self.robot.y
            dist = np.hypot(dx, dy)

            if dist < 0.3:
                self.stop()
                if not self.is_waiting:
                    self.is_waiting = True
                    self.wait_start_time = time.time()
                    self.collision_label.config(text="Arrived! Serving...")
                else:
                    if time.time() - self.wait_start_time > self.wait_duration:
                        self.is_waiting = False
                        self.destination = None
                        if self.destination_marker:
                            self.destination_marker.remove()
                            self.destination_marker = None
                        if not self.destination_queue:
                            self.collision_label.config(text="Done!")
            else:
                # DWA tính vận tốc
                v_l, v_r = self.planner.plan(goal=self.destination)
                self.robot.v_l = v_l
                self.robot.v_r = v_r

                # Thêm nhiễu vật lý (chỉ khi auto)
                self.robot.v_l += np.random.normal(0, 0.1)
                self.robot.v_r += np.random.normal(0, 0.1)

        # --- 2. LOGIC CẬP NHẬT VẬT LÝ (DÙNG CHUNG CHO CẢ AUTO VÀ MANUAL) ---
        # Đoạn này đã được đưa ra ngoài 'if auto_mode', nên nút bấm sẽ hoạt động

        # Chỉ cập nhật nếu robot đang có vận tốc (đang chạy)
        if self.robot.v_l != 0 or self.robot.v_r != 0:
            # Tính vị trí dự kiến để check va chạm
            v = (self.robot.v_l + self.robot.v_r) / 2
            omega = (self.robot.v_r - self.robot.v_l) / self.robot.L
            x_next = self.robot.x + v * np.cos(self.robot.theta) * self.dt
            y_next = self.robot.y + v * np.sin(self.robot.theta) * self.dt
            theta_next = self.robot.theta + omega * self.dt
            theta_next = (theta_next + np.pi) % (2 * np.pi) - np.pi  # Normalize

            if self.check_collision(x_next, y_next, theta_next):
                self.stop()
                self.collision_label.config(text="Collision Detected!")
            else:
                # Nếu không va chạm thì mới cập nhật vị trí thật
                self.robot.update_pose(self.dt)
                self.x_data.append(self.robot.x)
                self.y_data.append(self.robot.y)

        # --- 3. CẬP NHẬT ĐỒ HỌA ---
        self.path_line.set_data(self.x_data, self.y_data)
        self.robot_dot.set_data([self.robot.x], [self.robot.y])
        self.robot.update_graphics(self.ax)
        self.canvas.draw_idle()
        return (self.path_line,)

    # --- INPUT HANDLERS ---
    def move_forward(self):
        self.robot.v_l = self.robot.v_r = 1.5
        self.auto_mode = False

    def move_backward(self):
        self.robot.v_l = self.robot.v_r = -1.5
        self.auto_mode = False

    def turn_left(self):
        self.robot.v_l = 0.5
        self.robot.v_r = 1.5
        self.auto_mode = False

    def turn_right(self):
        self.robot.v_l = 1.5
        self.robot.v_r = 0.5
        self.auto_mode = False

    def stop(self):
        self.robot.v_l = self.robot.v_r = 0.0

    def reset(self):
        self.auto_mode = False
        self.destination_queue = []
        self.destination = None
        self.is_waiting = False
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
        elif event.keysym.lower() == "r":
            self.reset()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        # Thêm điểm mới vào hàng đợi
        target = (event.xdata, event.ydata)
        self.destination_queue.append(target)
        self.auto_mode = True
        print(f"Added to queue: {target}. Queue len: {len(self.destination_queue)}")
        # Vẽ dấu * đánh dấu đã nhận
        self.ax.plot(target[0], target[1], "r*", markersize=10, alpha=0.5)
        self.canvas.draw_idle()
