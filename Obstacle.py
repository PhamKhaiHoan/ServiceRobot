import numpy as np
import matplotlib.patches as patches


class Obstacle:
    def __init__(self):
        self.obs = []

    def add(self, new_obs):
        self.obs.append(new_obs)

    def clear(self):
        self.obs.clear()

    def remove(self, obs_to_remove):
        if obs_to_remove in self.obs:
            self.obs.remove(obs_to_remove)

    def generate_random_obstacles(self, count=5, area=(-10, 10)):
        for _ in range(count):
            x, y = np.random.uniform(*area, size=2)
            w, h = np.random.uniform(1.0, 3.0, size=2)
            self.add(patches.Rectangle((x, y), w, h, color="gray"))

    def label_obstacles(self, ax):
        for i, obs in enumerate(self.obs):
            if isinstance(obs, patches.Rectangle):
                x, y = obs.get_xy()
                ax.text(x, y, f"Obs {i}", fontsize=8, color="red")
            elif isinstance(obs, patches.Circle):
                x, y = obs.center
                ax.text(x, y, f"Obs {i}", fontsize=8, color="red")
