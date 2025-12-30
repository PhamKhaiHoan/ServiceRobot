import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import matplotlib.transforms as transforms


class Differential_Robot:
    def __init__(self, L, robot_length, robot_width, wheel_length, wheel_width):
        self.L = L
        self.robot_length = robot_length
        self.robot_width = robot_width
        self.wheel_length = wheel_length
        self.wheel_width = wheel_width

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v_l = 0.0
        self.v_r = 0.0
        self.v = 0.0
        self.omega = 0.0

        self.x_data = []
        self.y_data = []
        self.theta_data = []

        self.robot_body = patches.Rectangle(
            (-self.robot_length / 2, -self.robot_width / 2),
            self.robot_length,
            self.robot_width,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
        self.left_wheel = patches.Rectangle(
            (-self.wheel_length / 2, -self.wheel_width / 2),
            self.wheel_length,
            self.wheel_width,
            color="black",
        )
        self.right_wheel = patches.Rectangle(
            (-self.wheel_length / 2, -self.wheel_width / 2),
            self.wheel_length,
            self.wheel_width,
            color="black",
        )

        self.heading_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle="->", color="red", mutation_scale=15
        )
        self.left_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle="->", color="green", mutation_scale=10
        )
        self.right_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle="->", color="orange", mutation_scale=10
        )

    def update_pose(self, dt):
        self.v = (self.v_l + self.v_r) / 2
        self.omega = (self.v_r - self.v_l) / self.L
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.omega * dt
        self.theta = self.normalize_angle(self.theta)

        self.x_data.append(self.x)
        self.y_data.append(self.y)
        self.theta_data.append(self.theta)

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v_l = 0.0
        self.v_r = 0.0
        self.v = 0.0
        self.omega = 0.0
        self.x_data.clear()
        self.y_data.clear()
        self.theta_data.clear()

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def update_graphics(self, ax):
        dx = np.cos(self.theta)
        dy = np.sin(self.theta)
        perp_dx = -np.sin(self.theta)
        perp_dy = np.cos(self.theta)
        half_L = self.L / 2

        body_transform = (
            transforms.Affine2D().rotate(self.theta).translate(self.x, self.y)
            + ax.transData
        )
        self.robot_body.set_transform(body_transform)

        left_x = self.x + half_L * perp_dx
        left_y = self.y + half_L * perp_dy
        right_x = self.x - half_L * perp_dx
        right_y = self.y - half_L * perp_dy

        left_transform = (
            transforms.Affine2D().rotate(self.theta).translate(left_x, left_y)
            + ax.transData
        )
        right_transform = (
            transforms.Affine2D().rotate(self.theta).translate(right_x, right_y)
            + ax.transData
        )

        self.left_wheel.set_transform(left_transform)
        self.right_wheel.set_transform(right_transform)

        arrow_scale = 1.0
        self.heading_arrow.set_positions(
            (self.x, self.y), (self.x + arrow_scale * dx, self.y + arrow_scale * dy)
        )
        self.left_arrow.set_positions(
            (left_x, left_y),
            (
                left_x + arrow_scale * self.v_l * dx,
                left_y + arrow_scale * self.v_l * dy,
            ),
        )
        self.right_arrow.set_positions(
            (right_x, right_y),
            (
                right_x + arrow_scale * self.v_r * dx,
                right_y + arrow_scale * self.v_r * dy,
            ),
        )

    def get_robot_corners(self, x, y, theta):
        half_l = self.robot_length / 2
        half_w = self.robot_width / 2
        corners = np.array(
            [[-half_l, -half_w], [-half_l, half_w], [half_l, half_w], [half_l, -half_w]]
        )
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return (R @ corners.T).T + np.array([x, y])
