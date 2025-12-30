import numpy as np
import heapq
import matplotlib.patches as patches


class AStarPlanner:
    def __init__(
        self,
        obstacles,
        resolution=0.5,
        robot_radius=0.8,
        map_bounds=(-20, 20, -20, 20),
    ):
        """
        resolution    : grid size
        robot_radius  : inflate obstacle
        map_bounds    : xmin, xmax, ymin, ymax
        """
        self.obstacles = obstacles
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.xmin, self.xmax, self.ymin, self.ymax = map_bounds

        self.x_width = int((self.xmax - self.xmin) / resolution)
        self.y_width = int((self.ymax - self.ymin) / resolution)

        self.ox, self.oy = self._build_obstacle_map()

    # =====================================================
    # ===================== PLANNER =======================
    # =====================================================
    def plan(self, start, goal):
        sx, sy = self._world_to_grid(start)
        gx, gy = self._world_to_grid(goal)

        open_set = []
        heapq.heappush(open_set, (0, (sx, sy)))
        came_from = {}
        g_cost = {(sx, sy): 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == (gx, gy):
                return self._reconstruct_path(came_from, current)

            for nx, ny in self._neighbors(current):
                new_cost = g_cost[current] + self._distance(current, (nx, ny))
                if (nx, ny) not in g_cost or new_cost < g_cost[(nx, ny)]:
                    g_cost[(nx, ny)] = new_cost
                    f_cost = new_cost + self._heuristic((nx, ny), (gx, gy))
                    heapq.heappush(open_set, (f_cost, (nx, ny)))
                    came_from[(nx, ny)] = current

        print("A* failed to find path")
        return []

    # =====================================================
    # ===================== MAP ===========================
    # =====================================================
    def _build_obstacle_map(self):
        ox, oy = [], []

        for obs in self.obstacles.obs:
            if isinstance(obs, patches.Circle):
                cx, cy = obs.center
                r = obs.radius + self.robot_radius
                for angle in np.linspace(0, 2 * np.pi, 36):
                    ox.append(cx + r * np.cos(angle))
                    oy.append(cy + r * np.sin(angle))

            elif isinstance(obs, patches.Rectangle):
                x, y = obs.get_xy()
                w, h = obs.get_width(), obs.get_height()
                x -= self.robot_radius
                y -= self.robot_radius
                w += 2 * self.robot_radius
                h += 2 * self.robot_radius

                for i in np.arange(x, x + w, self.resolution):
                    ox.append(i)
                    oy.append(y)
                    ox.append(i)
                    oy.append(y + h)
                for j in np.arange(y, y + h, self.resolution):
                    ox.append(x)
                    oy.append(j)
                    ox.append(x + w)
                    oy.append(j)

        return ox, oy

    # =====================================================
    # ===================== GRID ==========================
    # =====================================================
    def _world_to_grid(self, pos):
        gx = int((pos[0] - self.xmin) / self.resolution)
        gy = int((pos[1] - self.ymin) / self.resolution)
        return gx, gy

    def _grid_to_world(self, node):
        x = node[0] * self.resolution + self.xmin
        y = node[1] * self.resolution + self.ymin
        return (x, y)

    def _is_valid(self, node):
        x, y = node
        if x < 0 or y < 0 or x >= self.x_width or y >= self.y_width:
            return False

        wx, wy = self._grid_to_world(node)
        for ox, oy in zip(self.ox, self.oy):
            if np.hypot(wx - ox, wy - oy) <= self.resolution:
                return False

        return True

    # =====================================================
    # ===================== A* ============================
    # =====================================================
    def _neighbors(self, node):
        moves = [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
        result = []
        for dx, dy in moves:
            nxt = (node[0] + dx, node[1] + dy)
            if self._is_valid(nxt):
                result.append(nxt)
        return result

    def _heuristic(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def _distance(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    # =====================================================
    # ===================== PATH ==========================
    # =====================================================
    def _reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(self._grid_to_world(current))
            current = came_from[current]
        path.reverse()
        return path
