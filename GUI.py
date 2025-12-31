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
from Robot import LIDAR


class RobotSimulatorGUI:
    def __init__(
        self, root, robot, obstacles, locations, dt=0.1, customer_manager=None
    ):
        self.root = root
        self.robot = robot
        self.obstacles = obstacles
        self.locations = locations
        self.dt = dt
        self.customer_manager = customer_manager  # Quản lý khách hàng động

        # --- TRẠNG THÁI HỆ THỐNG ---
        self.destination_queue = []
        self.task_names = []
        self.is_executing = False
        self.is_waiting = False
        self.wait_start_time = 0
        self.wait_duration = 0.5  # Thời gian chờ (nhanh)

        self.destination = None
        self.current_algo = "DWA"

        # --- AUTO RETURN TO KITCHEN ---
        self.auto_return_countdown = 3.0  # 3 giây đếm ngược
        self.idle_start_time = None  # Thời điểm bắt đầu idle
        self.is_counting_down = False  # Đang đếm ngược?

        # --- LIDAR RADAR ---
        self.lidar = LIDAR(num_beams=36, max_range=4.0, fov=360)
        self.show_lidar = True  # Bật/tắt hiển thị LIDAR

        # --- KHỞI TẠO PLANNERS & CẤU HÌNH TỐC ĐỘ ---
        self.dwa = DWA_Planner(robot, obstacles, dt, predict_time=2.0)
        self.dwa.max_speed = 4.0
        self.dwa.max_accel = 3.0
        self.dwa.max_yawrate = 6.0

        self.bug = BugPlanner(robot, obstacles, step_size=0.5)
        self.astar = AStarPlanner(obstacles, resolution=0.5, robot_radius=1.0)
        self.dijkstra = DijkstraPlanner(obstacles, resolution=0.5, robot_radius=1.0)

        self.global_path = []
        self.current_wp_index = 0

        # --- ĐỒ HỌA CHÍNH ---
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-13, 16)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor("#F5F5DC")  # Màu nền beige

        for obs in self.obstacles.obs:
            self.ax.add_patch(obs)

        # Thêm khách hàng động vào đồ họa
        if self.customer_manager:
            self.customer_manager.add_to_axes(self.ax)

        # Vị trí thực của bàn (để hiển thị label trên mặt bàn)
        TABLE_POSITIONS = [
            (-8, 5, "Table 1"),
            (8, 5, "Table 2"),
            (-8, -1, "Table 3"),
            (8, -1, "Table 4"),
            (-8, -7, "Table 5"),
            (8, -7, "Table 6"),
        ]

        # Hiển thị label Table trực tiếp trên mặt bàn
        for tx, ty, tname in TABLE_POSITIONS:
            self.ax.text(
                tx,
                ty,
                tname,
                color="white",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#8B4513",
                    alpha=0.9,
                    edgecolor="none",
                ),
            )

        # Kitchen label
        self.ax.text(
            0,
            11.5,
            "KITCHEN",
            color="white",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#333333",
                edgecolor="white",
                linewidth=2,
            ),
        )

        # Hiển thị điểm đích (robot sẽ dừng ở đây, cạnh bàn)
        for name, (x, y) in self.locations.items():
            if name != "Kitchen":
                self.ax.plot(x, y, "go", markersize=6, alpha=0.5)  # Điểm dừng

        self.robot_body = self.ax.add_patch(self.robot.robot_body)
        self.left_wheel = self.ax.add_patch(self.robot.left_wheel)
        self.right_wheel = self.ax.add_patch(self.robot.right_wheel)
        # self.heading_arrow = self.ax.add_patch(self.robot.heading_arrow)  # Da xoa mui ten
        (self.path_line,) = self.ax.plot([], [], "r-", linewidth=1, alpha=0.5)
        (self.global_path_line,) = self.ax.plot([], [], "g--", linewidth=1.5, alpha=0.7)
        (self.destination_marker,) = self.ax.plot([], [], "r*", markersize=12)
        (self.queue_markers,) = self.ax.plot(
            [], [], "r*", markersize=6, alpha=0.3
        )  # Điểm trong queue (nhỏ, mờ)

        # LIDAR visualization trên map chính
        self.lidar_lines = []  # Các đường tia quét
        for _ in range(self.lidar.num_beams):
            (line,) = self.ax.plot([], [], "c-", linewidth=0.5, alpha=0.3)
            self.lidar_lines.append(line)
        (self.lidar_hits,) = self.ax.plot(
            [], [], "ro", markersize=3, alpha=0.7
        )  # Điểm phát hiện

        self.x_data, self.y_data = [], []

        # Stuck detection
        self.stuck_counter = 0
        self.last_position = (self.robot.x, self.robot.y)

        # Manual control mode
        self.manual_mode = False
        self.manual_v = 0
        self.manual_omega = 0

        # --- RADAR DISPLAY RIÊNG (kiểu Cartesian như hình gốc) ---
        self.fig_radar, self.ax_radar = plt.subplots(figsize=(3.2, 3.2))
        self.fig_radar.patch.set_facecolor("#1a1a2e")  # Nền figure
        self.ax_radar.set_xlim(-5, 5)
        self.ax_radar.set_ylim(-5, 5)
        self.ax_radar.set_aspect("equal")
        self.ax_radar.set_facecolor("#16213e")  # Nền đậm
        self.ax_radar.tick_params(colors="#888888", labelsize=6)
        self.ax_radar.grid(True, color="#333333", alpha=0.5, linewidth=0.5)
        self.ax_radar.set_xlabel("X (m)", color="#888888", fontsize=7)
        self.ax_radar.set_ylabel("Y (m)", color="#888888", fontsize=7)
        self.ax_radar.set_title(
            "LIDAR Map", color="#00ff88", fontsize=10, fontweight="bold"
        )

        # Robot marker ở trung tâm
        self.radar_robot = self.ax_radar.plot(
            0, 0, "s", color="cyan", markersize=8, label="Robot"
        )[0]

        # Scatter cho các điểm LIDAR phát hiện
        self.radar_scatter = self.ax_radar.scatter(
            [], [], c="lime", s=15, alpha=0.9, marker=".", label="Detected"
        )

        # Legend
        self.ax_radar.legend(
            loc="upper right",
            fontsize=6,
            facecolor="#1a1a2e",
            labelcolor="white",
            framealpha=0.8,
        )

        self.fig_radar.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(
            row=0, column=0, rowspan=20, sticky="nsew", padx=5, pady=5
        )

        # Click to move - kết nối sự kiện click chuột
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        self._create_controls()

        self.ani = FuncAnimation(
            self.fig, self.update, interval=80, blit=False, cache_frame_data=False
        )

    def _create_controls(self):
        # Main panel bên phải
        panel = ttk.Frame(self.root)
        panel.grid(row=0, column=1, sticky="ns", padx=10, pady=10)

        # Logic Mode
        ttk.Label(panel, text="🔧 LOGIC MODE", font=("Arial", 10, "bold")).pack(
            pady=(0, 5)
        )
        self.algo_var = tk.StringVar(value="DWA")
        frame_algo = ttk.Frame(panel)
        frame_algo.pack()
        for alg in ["DWA", "BUG", "A_STAR", "DIJKSTRA"]:
            ttk.Radiobutton(
                frame_algo,
                text=alg,
                variable=self.algo_var,
                value=alg,
                command=self.change_algo,
            ).pack(anchor="w")

        # Select Tables
        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(panel, text="🍽️ SELECT TABLES", font=("Arial", 10, "bold")).pack()
        frame_tables = ttk.Frame(panel)
        frame_tables.pack(pady=5)

        i = 0
        sorted_locs = sorted([k for k in self.locations.keys() if k != "Kitchen"])
        for name in sorted_locs:
            btn = ttk.Button(
                frame_tables,
                text=f"+ {name}",
                command=lambda n=name: self.add_to_queue(n),
                width=10,
            )
            btn.grid(row=i // 2, column=i % 2, padx=2, pady=2)
            i += 1

        # --- QUEUE DISPLAY (Cải thiện) ---
        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(panel, text="📋 DELIVERY QUEUE", font=("Arial", 10, "bold")).pack()

        # Frame cho queue với border
        queue_frame = ttk.Frame(panel, relief="sunken", borderwidth=2)
        queue_frame.pack(fill="x", pady=5, padx=5)

        # Listbox hiển thị queue
        self.queue_listbox = tk.Listbox(
            queue_frame,
            height=4,
            font=("Consolas", 9),
            bg="#2d2d2d",
            fg="#00ff00",
            selectbackground="#005500",
            borderwidth=0,
            highlightthickness=0,
        )
        self.queue_listbox.pack(fill="x", padx=2, pady=2)

        # Label đếm số lượng
        self.lbl_queue_count = ttk.Label(
            panel, text="Items: 0", foreground="gray", font=("Arial", 9)
        )
        self.lbl_queue_count.pack()

        # Control Buttons
        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=5)

        # Style cho nút Start
        style = ttk.Style()
        style.configure("Start.TButton", font=("Arial", 10, "bold"))
        style.configure(
            "Emergency.TButton", foreground="red", font=("Arial", 10, "bold")
        )

        ttk.Button(
            panel,
            text="▶ START SERVICE",
            command=self.start_service,
            style="Start.TButton",
        ).pack(fill="x", pady=2)

        ttk.Button(panel, text="✖ CLEAR QUEUE", command=self.clear_queue).pack(
            fill="x", pady=2
        )

        ttk.Button(panel, text="🔄 RESET SYSTEM", command=self.reset_system).pack(
            fill="x", pady=2
        )

        ttk.Button(
            panel,
            text="🏠 RETURN KITCHEN",
            style="Emergency.TButton",
            command=self.return_to_kitchen,
        ).pack(fill="x", pady=5)

        # --- RADAR DISPLAY ---
        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=5)
        ttk.Label(panel, text="📡 RADAR DISPLAY", font=("Arial", 10, "bold")).pack(
            pady=(5, 2)
        )

        # Embed radar figure
        self.canvas_radar = FigureCanvasTkAgg(self.fig_radar, master=panel)
        self.canvas_radar.get_tk_widget().pack(fill="x", pady=5)

        self.lidar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            panel,
            text="Show LIDAR on Map",
            variable=self.lidar_var,
            command=self._toggle_lidar,
        ).pack(anchor="w")

        # --- MANUAL CONTROL PANEL ---
        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=5)
        ttk.Label(panel, text="MANUAL CONTROL", font=("Arial", 10, "bold")).pack(
            pady=(5, 2)
        )

        # Toggle Manual Mode
        self.manual_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            panel,
            text="Enable Manual Mode",
            variable=self.manual_var,
            command=self._toggle_manual_mode,
        ).pack(pady=2)

        # Direction buttons frame
        frame_manual = ttk.Frame(panel)
        frame_manual.pack(pady=5)

        # Arrow buttons layout:
        #       [↑]
        # [←]  [■]  [→]
        #       [↓]
        self.btn_up = ttk.Button(frame_manual, text="↑", width=3)
        self.btn_up.grid(row=0, column=1, padx=2, pady=2)

        self.btn_left = ttk.Button(frame_manual, text="←", width=3)
        self.btn_left.grid(row=1, column=0, padx=2, pady=2)

        self.btn_stop = ttk.Button(
            frame_manual, text="■", width=3, command=self._manual_stop
        )
        self.btn_stop.grid(row=1, column=1, padx=2, pady=2)

        self.btn_right = ttk.Button(frame_manual, text="→", width=3)
        self.btn_right.grid(row=1, column=2, padx=2, pady=2)

        self.btn_down = ttk.Button(frame_manual, text="↓", width=3)
        self.btn_down.grid(row=2, column=1, padx=2, pady=2)

        # Bind button press/release events
        self.btn_up.bind("<ButtonPress-1>", lambda e: self._manual_move("up"))
        self.btn_up.bind("<ButtonRelease-1>", lambda e: self._manual_stop())
        self.btn_down.bind("<ButtonPress-1>", lambda e: self._manual_move("down"))
        self.btn_down.bind("<ButtonRelease-1>", lambda e: self._manual_stop())
        self.btn_left.bind("<ButtonPress-1>", lambda e: self._manual_move("left"))
        self.btn_left.bind("<ButtonRelease-1>", lambda e: self._manual_stop())
        self.btn_right.bind("<ButtonPress-1>", lambda e: self._manual_move("right"))
        self.btn_right.bind("<ButtonRelease-1>", lambda e: self._manual_stop())

        # Click instruction
        ttk.Label(
            panel, text="💡 Click on map to move", font=("Arial", 8), foreground="gray"
        ).pack(pady=2)

        self.lbl_status = ttk.Label(
            panel, text="Status: Idle", foreground="blue", font=("Arial", 11, "bold")
        )
        self.lbl_status.pack(pady=10)

    # --- [NEW] HÀM RESET TOÀN BỘ ---
    def reset_system(self):
        print("--- SYSTEM RESET ---")
        self.stop_robot()

        # 1. Reset logic
        self.is_executing = False
        self.destination_queue = []
        self.task_names = []
        self.destination = None
        self.global_path = []
        self.is_waiting = False
        self.current_wp_index = 0
        self.stuck_counter = 0
        self._cancel_countdown()  # Hủy đếm ngược

        # 2. Reset vị trí Robot về Bếp
        if "Kitchen" in self.locations:
            self.robot.x, self.robot.y = self.locations["Kitchen"]
            self.robot.theta = -1.57  # Hướng xuống dưới
            self.robot.v_l = 0
            self.robot.v_r = 0
            self.last_position = (self.robot.x, self.robot.y)

        # 3. Xóa dữ liệu vẽ đường
        self.x_data = []
        self.y_data = []
        self.path_line.set_data([], [])
        self.global_path_line.set_data([], [])
        self.destination_marker.set_data([], [])
        self.queue_markers.set_data([], [])  # Xóa các điểm queue

        # 4. Cập nhật GUI
        self.update_queue_label()
        self.lbl_status.config(text="Status: System Reset Done.")
        self.canvas.draw_idle()

    # --- CÁC HÀM KHÁC GIỮ NGUYÊN ---
    def return_to_kitchen(self):
        print("!!! INTERRUPT: RETURNING TO KITCHEN !!!")
        self.stop_robot()
        self._cancel_countdown()  # Hủy đếm ngược
        self.destination_queue = []
        self.task_names = []
        self.stuck_counter = 0
        if "Kitchen" in self.locations:
            self.destination_queue.append(self.locations["Kitchen"])
            self.task_names.append("Kitchen (Emergency)")

        self.destination = None
        self.global_path = []
        self.current_wp_index = 0
        self.is_waiting = False
        self.is_executing = True
        self.update_queue_label()
        self.lbl_status.config(text="Status: EMERGENCY RETURN!", foreground="red")

    def change_algo(self):
        self.current_algo = self.algo_var.get()
        self.global_path = []
        print(f"Algorithm changed to: {self.current_algo}")

    def add_to_queue(self, name):
        """Thêm bàn vào hàng đợi - CHỈ THÊM, KHÔNG TỰ ĐỘNG CHẠY"""
        if name in self.locations:
            self.destination_queue.append(self.locations[name])
            self.task_names.append(name)
            self._cancel_countdown()  # Hủy đếm ngược khi có yêu cầu mới
            self.update_queue_label()
            self.lbl_status.config(text=f"Added {name} to queue")

    def update_queue_label(self):
        """Cập nhật hiển thị queue trong Listbox"""
        # Xóa listbox cũ
        self.queue_listbox.delete(0, tk.END)

        if self.task_names:
            for i, name in enumerate(self.task_names):
                self.queue_listbox.insert(tk.END, f" {i+1}. {name}")
            self.lbl_queue_count.config(
                text=f"Items: {len(self.task_names)}", foreground="green"
            )
        else:
            self.queue_listbox.insert(tk.END, " (Empty)")
            self.lbl_queue_count.config(text="Items: 0", foreground="gray")

    def clear_queue(self):
        self.destination_queue = []
        self.task_names = []
        self.is_executing = False
        self.destination = None
        self.global_path = []
        self.current_wp_index = 0
        self.stuck_counter = 0
        self._cancel_countdown()  # Hủy đếm ngược
        self.global_path_line.set_data([], [])
        self.destination_marker.set_data([], [])
        self.queue_markers.set_data([], [])  # Xóa các điểm queue
        self.update_queue_label()
        self.lbl_status.config(text="Status: Queue Cleared")

    def start_service(self):
        if not self.destination_queue:
            self.lbl_status.config(text="Status: Queue Empty!")
            return
        self._cancel_countdown()  # Hủy đếm ngược khi bắt đầu service
        # Không tự động thêm Kitchen - sẽ tự động quay về sau 3 giây khi hết queue
        self.is_executing = True
        self.lbl_status.config(text="Status: Executing...")

    def update(self, frame):
        if self.is_executing:
            if self.destination is None and self.destination_queue:
                self.destination = self.destination_queue.pop(0)
                current_task = self.task_names.pop(0)
                self.update_queue_label()
                self.lbl_status.config(text=f"Going to: {current_task}")
                self.global_path = []
                self.current_wp_index = 0

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
                        print(
                            f"⚠️ Cảnh báo: {self.current_algo} không tìm được đường! Chuyển sang DWA tạm thời."
                        )
                        self.global_path_line.set_data([], [])

            if self.destination is not None:
                self.destination_marker.set_data(
                    [self.destination[0]], [self.destination[1]]
                )
                # Hiển thị các điểm trong queue (chưa đến lượt)
                if self.destination_queue:
                    qx = [p[0] for p in self.destination_queue]
                    qy = [p[1] for p in self.destination_queue]
                    self.queue_markers.set_data(qx, qy)
                else:
                    self.queue_markers.set_data([], [])

                dist = np.hypot(
                    self.destination[0] - self.robot.x,
                    self.destination[1] - self.robot.y,
                )

                if dist < 0.6:  # Tang nguong de khop voi DWA
                    self.stop_robot()
                    if not self.is_waiting:
                        self.is_waiting = True
                        self.wait_start_time = time.time()
                        self.lbl_status.config(text="Arrived! Serving...")
                    elif time.time() - self.wait_start_time > self.wait_duration:
                        self.is_waiting = False
                        self.destination = None
                        self.global_path = []  # Clear path khi đến nơi
                        self.current_wp_index = 0  # Reset waypoint index
                        self.global_path_line.set_data([], [])  # Clear đường vẽ
                        if not self.destination_queue:
                            self.is_executing = False
                            # Bắt đầu đếm ngược để quay về bếp
                            self._start_idle_countdown()
                        else:
                            self.lbl_status.config(text="Status: Moving to next...")

                elif not self.is_waiting:
                    v_l, v_r = 0, 0
                    use_global_path = (
                        self.current_algo in ["A_STAR", "DIJKSTRA"]
                        and self.global_path
                        and len(self.global_path) > 0
                    )

                    # Cập nhật vị trí khách hàng động cho các planner
                    if self.customer_manager:
                        dynamic_obs = self.customer_manager.get_positions()
                        self.dwa.set_dynamic_obstacles(dynamic_obs)
                        self.bug.set_dynamic_obstacles(dynamic_obs)

                    try:
                        if use_global_path:
                            if self.current_wp_index < len(self.global_path):
                                target_wp = self.global_path[self.current_wp_index]
                                dist_wp = np.hypot(
                                    target_wp[0] - self.robot.x,
                                    target_wp[1] - self.robot.y,
                                )
                                if dist_wp < 0.6:
                                    self.current_wp_index += 1

                                idx = min(
                                    self.current_wp_index, len(self.global_path) - 1
                                )
                                drive_target = self.global_path[idx]
                                v_l, v_r = self.dwa.plan(drive_target)
                            else:
                                # Đã đi hết waypoints, dùng DWA đến đích cuối
                                v_l, v_r = self.dwa.plan(self.destination)
                        else:
                            # Fallback: dùng DWA hoặc BUG khi không có global path
                            if self.current_algo == "BUG":
                                v_l, v_r = self.bug.compute_velocity(self.destination)
                            else:
                                v_l, v_r = self.dwa.plan(self.destination)

                        # Hiển thị trạng thái blocked nếu planner báo bị chặn
                        if self.current_algo == "BUG" and self.bug.is_blocked:
                            self.lbl_status.config(text="Status: Avoiding obstacle...")
                        elif hasattr(self.dwa, "is_blocked") and self.dwa.is_blocked:
                            self.lbl_status.config(
                                text="Status: Path blocked, finding route..."
                            )

                    except Exception as e:
                        print(f"Planner error: {e}")
                        v_l, v_r = 0, 0

                    # Stuck detection
                    current_pos = (self.robot.x, self.robot.y)
                    movement = np.hypot(
                        current_pos[0] - self.last_position[0],
                        current_pos[1] - self.last_position[1],
                    )
                    if movement < 0.01 and (abs(v_l) > 0.1 or abs(v_r) > 0.1):
                        self.stuck_counter += 1
                        if self.stuck_counter > 50:  # Stuck for too long
                            print("Robot stuck! Attempting recovery...")
                            # Recovery: back up and turn
                            v_l, v_r = -1.0, -0.5
                            self.stuck_counter = 0
                    else:
                        self.stuck_counter = 0
                    self.last_position = current_pos

                    self.robot.v_l = v_l
                    self.robot.v_r = v_r

            self.robot.update_pose(self.dt)
            self.x_data.append(self.robot.x)
            self.y_data.append(self.robot.y)

        # Manual mode - cập nhật vị trí robot khi điều khiển thủ công
        elif self.manual_mode:
            self.robot.update_pose(self.dt)
            self.x_data.append(self.robot.x)
            self.y_data.append(self.robot.y)

        # Limit path data to prevent memory issues
        max_path_points = 1000
        if len(self.x_data) > max_path_points:
            self.x_data = self.x_data[-max_path_points:]
            self.y_data = self.y_data[-max_path_points:]

        self.path_line.set_data(self.x_data, self.y_data)
        self.robot.update_graphics(self.ax)

        # --- Cập nhật LIDAR radar ---
        self._update_lidar()

        # --- Cập nhật khách hàng động ---
        if self.customer_manager:
            self.customer_manager.update(self.dt, self.obstacles.obs, robot=self.robot)

        # Cập nhật queue markers khi không đang thực thi
        if not self.is_executing and not self.destination:
            if self.destination_queue:
                qx = [p[0] for p in self.destination_queue]
                qy = [p[1] for p in self.destination_queue]
                self.queue_markers.set_data(qx, qy)
            else:
                self.queue_markers.set_data([], [])

        # --- Xử lý đếm ngược tự động quay về bếp ---
        self._update_idle_countdown()

        try:
            self.canvas.draw_idle()
        except Exception:
            pass  # Ignore drawing errors during window close

        return [
            self.robot_body,
            self.left_wheel,
            self.right_wheel,
            self.path_line,
            self.global_path_line,
            self.destination_marker,
        ]

    def _start_idle_countdown(self):
        """Bắt đầu đếm ngược khi robot hoàn thành nhiệm vụ"""
        kitchen_pos = self.locations.get("Kitchen", (0, 10))
        dist_to_kitchen = np.hypot(
            kitchen_pos[0] - self.robot.x, kitchen_pos[1] - self.robot.y
        )

        # Chỉ đếm ngược nếu robot không ở bếp
        if dist_to_kitchen > 0.5:
            self.is_counting_down = True
            self.idle_start_time = time.time()
            self.lbl_status.config(text="Idle - Returning in 3s...")
        else:
            self.is_counting_down = False
            self.lbl_status.config(text="Status: At Kitchen/Idle")

    def _update_idle_countdown(self):
        """Cập nhật đếm ngược và tự động quay về bếp"""
        if self.is_counting_down and self.idle_start_time is not None:
            elapsed = time.time() - self.idle_start_time
            remaining = self.auto_return_countdown - elapsed

            if remaining > 0:
                # Hiển thị đếm ngược
                self.lbl_status.config(text=f"Idle - Returning in {remaining:.1f}s...")
            else:
                # Hết thời gian, quay về bếp
                self.is_counting_down = False
                self.idle_start_time = None
                self._auto_return_to_kitchen()

    def _cancel_countdown(self):
        """Hủy đếm ngược khi có yêu cầu mới"""
        self.is_counting_down = False
        self.idle_start_time = None

    def _auto_return_to_kitchen(self):
        """Tự động quay về bếp sau khi đếm ngược xong"""
        if "Kitchen" in self.locations:
            # Kiểm tra đã ở bếp chưa
            kitchen_pos = self.locations["Kitchen"]
            dist = np.hypot(
                kitchen_pos[0] - self.robot.x, kitchen_pos[1] - self.robot.y
            )
            if dist > 0.5:  # Chưa ở bếp
                print("Auto returning to Kitchen...")
                self.destination_queue = [kitchen_pos]
                self.task_names = ["Kitchen (Auto)"]
                self.global_path = []
                self.current_wp_index = 0
                self.is_executing = True
                self.update_queue_label()
                self.lbl_status.config(text="Auto returning to Kitchen...")

    def stop_robot(self):
        self.robot.v_l = 0
        self.robot.v_r = 0

    # --- LIDAR FUNCTIONS ---
    def _toggle_lidar(self):
        """Bật/tắt hiển thị LIDAR"""
        self.show_lidar = self.lidar_var.get()
        if not self.show_lidar:
            # Ẩn các tia LIDAR
            for line in self.lidar_lines:
                line.set_data([], [])
            self.lidar_hits.set_data([], [])

    def _update_lidar(self):
        """Cập nhật và vẽ LIDAR"""
        # Lấy vị trí vật cản động
        dynamic_obs = []
        if self.customer_manager:
            dynamic_obs = self.customer_manager.get_positions()

        # Quét LIDAR
        self.lidar.scan(
            self.robot.x,
            self.robot.y,
            self.robot.theta,
            self.obstacles.obs,
            dynamic_obs,
        )

        # Cập nhật RADAR display (luôn cập nhật)
        self._update_radar_display()

        # Vẽ LIDAR trên map chính (chỉ khi bật)
        if self.show_lidar:
            scan_points = self.lidar.get_scan_points(
                self.robot.x, self.robot.y, self.robot.theta
            )

            for i, (end_x, end_y) in enumerate(scan_points):
                self.lidar_lines[i].set_data(
                    [self.robot.x, end_x], [self.robot.y, end_y]
                )
                # Đổi màu tia nếu phát hiện vật cản gần
                if self.lidar.ranges[i] < self.lidar.max_range * 0.8:
                    self.lidar_lines[i].set_color("yellow")
                    self.lidar_lines[i].set_alpha(0.5)
                else:
                    self.lidar_lines[i].set_color("cyan")
                    self.lidar_lines[i].set_alpha(0.2)

            # Vẽ điểm phát hiện vật cản
            if self.lidar.hit_points:
                hit_x = [p[0] for p in self.lidar.hit_points]
                hit_y = [p[1] for p in self.lidar.hit_points]
                self.lidar_hits.set_data(hit_x, hit_y)
            else:
                self.lidar_hits.set_data([], [])

    def _update_radar_display(self):
        """Cập nhật màn hình radar Cartesian (giống hình gốc)"""
        # Lấy các điểm LIDAR hit trong tọa độ tương đối với robot
        angles = self.lidar.angles
        ranges = self.lidar.ranges

        # Chuyển đổi sang tọa độ Cartesian (tương đối với robot)
        hit_x = []
        hit_y = []

        for i, r in enumerate(ranges):
            if r < self.lidar.max_range * 0.95:  # Có vật cản
                # Tọa độ tương đối với robot (robot ở tâm)
                x = r * np.cos(angles[i])
                y = r * np.sin(angles[i])
                hit_x.append(x)
                hit_y.append(y)

        if hit_x:
            self.radar_scatter.set_offsets(np.column_stack([hit_x, hit_y]))
        else:
            self.radar_scatter.set_offsets(np.empty((0, 2)))

        # Cập nhật canvas radar
        try:
            self.canvas_radar.draw_idle()
        except Exception:
            pass

    # --- MANUAL CONTROL FUNCTIONS ---
    def _toggle_manual_mode(self):
        """Bật/tắt chế độ điều khiển thủ công"""
        self.manual_mode = self.manual_var.get()
        if self.manual_mode:
            # Dừng các tác vụ tự động
            self.is_executing = False
            self._cancel_countdown()
            self.destination = None
            self.stop_robot()
            self.lbl_status.config(text="Manual Mode: ON", foreground="green")
        else:
            self._manual_stop()
            self.lbl_status.config(text="Manual Mode: OFF", foreground="blue")

    def _manual_move(self, direction):
        """Di chuyển robot theo hướng được chỉ định"""
        if not self.manual_mode:
            return

        speed = 3.0  # Tốc độ di chuyển
        turn_speed = 2.0  # Tốc độ xoay

        if direction == "up":
            # Đi thẳng về phía trước
            self.robot.v_l = speed
            self.robot.v_r = speed
        elif direction == "down":
            # Đi lùi
            self.robot.v_l = -speed
            self.robot.v_r = -speed
        elif direction == "left":
            # Xoay trái
            self.robot.v_l = -turn_speed
            self.robot.v_r = turn_speed
        elif direction == "right":
            # Xoay phải
            self.robot.v_l = turn_speed
            self.robot.v_r = -turn_speed

    def _manual_stop(self):
        """Dừng robot khi thả nút"""
        self.robot.v_l = 0
        self.robot.v_r = 0

    # --- CLICK TO MOVE FUNCTION ---
    def _on_canvas_click(self, event):
        """Xử lý khi click vào canvas - thêm vị trí vào hàng đợi"""
        if event.inaxes != self.ax:
            return  # Click ngoài vùng đồ họa

        if event.button != 1:  # Chỉ xử lý click chuột trái
            return

        click_x, click_y = event.xdata, event.ydata

        if click_x is None or click_y is None:
            return

        # Tắt manual mode nếu đang bật
        if self.manual_mode:
            self.manual_var.set(False)
            self.manual_mode = False

        # Hủy đếm ngược nếu có
        self._cancel_countdown()

        # Thêm vào hàng đợi (queue)
        print(f"Click added to queue: ({click_x:.1f}, {click_y:.1f})")

        self.destination_queue.append((click_x, click_y))
        self.task_names.append(f"({click_x:.1f}, {click_y:.1f})")
        self.update_queue_label()

        # Nếu chưa đang chạy, bắt đầu chạy
        if not self.is_executing:
            self.is_executing = True
            self.lbl_status.config(text=f"Going to ({click_x:.1f}, {click_y:.1f})...")
