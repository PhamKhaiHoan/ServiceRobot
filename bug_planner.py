import numpy as np
import matplotlib.patches as patches


class BugPlanner:
    """
    Bug2 algorithm (simplified for simulation)
    Global planner: generate waypoints
    Local avoidance is handled by DWA
    """

    def __init__(self, robot, obstacles, step_size=0.5):
        self.robot = robot
        self.obstacles = obstacles
        self.step_size = step_size

    def plan(self, start, goal):
        """
        Return list of waypoints from start to goal
        """
        path = []
        current = np.array(start)
        goal = np.array(goal)

        max_iter = 500
        it = 0

        while np.linalg.norm(goal - current) > self.step_size and it < max_iter:
            it += 1

            # Step toward goal
            direction = goal - current
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            next_point = current + self.step_size * direction

            # Check collision
            if not self.check_collision(next_point):
                path.append(tuple(next_point))
                current = next_point
            else:
                # Follow obstacle boundary
                boundary_path = self.follow_boundary(current, goal)
                path.extend(boundary_path)
                if boundary_path:
                    current = np.array(boundary_path[-1])
                else:
                    break

        path.append(tuple(goal))
        return path

    # ====================================================
    # ================= COLLISION ========================
    # ====================================================
    def check_collision(self, point):
        for obs in self.obstacles.obs:
            if isinstance(obs, patches.Circle):
                center = np.array(obs.center)
                if np.linalg.norm(point - center) <= obs.radius + 0.5:
                    return True
            elif isinstance(obs, patches.Rectangle):
                x, y = point
                bx, by = obs.get_xy()
                bw, bh = obs.get_width(), obs.get_height()
                if bx - 0.5 <= x <= bx + bw + 0.5 and by - 0.5 <= y <= by + bh + 0.5:
                    return True
        return False

    # ====================================================
    # ============ FOLLOW OBSTACLE =======================
    # ====================================================
    def follow_boundary(self, start, goal):
        """
        Simple left-hand rule boundary following
        """
        path = []
        pos = np.array(start)

        # Find obstacle
        obstacle = self.find_nearest_obstacle(pos)
        if obstacle is None:
            return path

        # Tangent directions
        angles = np.linspace(0, 2 * np.pi, 36)

        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            next_point = pos + self.step_size * direction

            if not self.check_collision(next_point):
                # Check if moving closer to goal
                if np.linalg.norm(goal - next_point) < np.linalg.norm(goal - pos):
                    path.append(tuple(next_point))
                    break

        return path

    def find_nearest_obstacle(self, point):
        min_dist = float("inf")
        nearest = None
        for obs in self.obstacles.obs:
            if isinstance(obs, patches.Circle):
                d = np.linalg.norm(point - np.array(obs.center)) - obs.radius
            elif isinstance(obs, patches.Rectangle):
                bx, by = obs.get_xy()
                bw, bh = obs.get_width(), obs.get_height()
                cx = np.clip(point[0], bx, bx + bw)
                cy = np.clip(point[1], by, by + bh)
                d = np.linalg.norm(point - np.array([cx, cy]))
            else:
                continue

            if d < min_dist:
                min_dist = d
                nearest = obs

        return nearest
