#!/usr/bin/env python3
import matplotlib.pyplot as plt
from math import pi, sqrt, atan2
from sympy import Matrix, symarray, sin, cos, eye, Expr, pprint, init_printing
import random
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa  # Ignore complaints about Axes3D not being used
init_printing()  # doctest: +SKIP


def rot_to_euler(r):
    sy = sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])

    singular = sy < 1e-6

    if not singular:
        x_euler = atan2(r[2, 1], r[2, 2])
        y_euler = atan2(-r[2, 0], sy)
        z_euler = atan2(r[1, 0], r[0, 0])
    else:
        x_euler = atan2(-r[1, 2], r[1, 1])
        y_euler = atan2(-r[2, 0], sy)
        z_euler = 0

    return x_euler, y_euler, z_euler


def check_intersect(x0, y0, z0, x1, y1, z1, x2, y2, z2):

    # set point that exists within a arm link can cause singularity
    ab = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1))
    ap = sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1))
    pb = sqrt((x2 - x0) * (x2 - x0) + (y2 - y0) * (y2 - y0) + (z2 - z0) * (z2 - z0))
    if abs(ab - (ap + pb)) < .05:
        return 1


class Joint:

    def __init__(self, dh):
        self.theta = 0
        self.target_theta = 0
        self.dh_theta = dh[0]
        self.dh_alpha = dh[1]
        self.dh_a = dh[2]
        self.dh_d = dh[3]
        self.transform_to_me = eye(4)

    def update_delta(self, delta):
        self.target_theta = delta + self.theta

    def update_theta(self):
        self.theta += float(self.target_theta - self.theta)

    def transform(self):
        transform = Matrix([[cos(self.dh_theta), -sin(self.dh_theta) * cos(self.dh_alpha),
                             sin(self.dh_theta) * sin(self.dh_alpha), self.dh_a * cos(self.dh_theta)],
                            [sin(self.dh_theta), cos(self.dh_theta) * cos(self.dh_alpha),
                             -cos(self.dh_theta) * sin(self.dh_alpha), self.dh_a * sin(self.dh_theta)],
                            [0, sin(self.dh_alpha), cos(self.dh_alpha), self.dh_d],
                            [0, 0, 0, 1]])
        return transform

    def solve_transform(self, theta_subs):
        transform_to_me = self.transform_to_me.xreplace(theta_subs)
        x_coord = transform_to_me[0, 3]
        y_coord = transform_to_me[1, 3]
        z_coord = transform_to_me[2, 3]
        (i_coord, j_coord, k_coord) = rot_to_euler(transform_to_me[0:3, 0:3])
        return Matrix([x_coord, y_coord, z_coord, i_coord, j_coord, k_coord])


class Armature:

    def __init__(self):
        self.link_list = []
        self.transform_list = []
        self.theta_list = []
        self.sym_theta_list = []
        self.paired_theta = []
        self.desiredxyzijk = Matrix([375, 175, 174, pi / 2, 0, 0])
        self.convergence_time = 0
        self.start_time = time.perf_counter()
        self.damping_coefficient = 0
        self.z_jacob = Matrix([])
    # add link to lists
    # two lists are maintained, one for dh transforms
    # other holds only joints with actuation

    def add_link(self, dh):
        new_joint = Joint(dh)
        if isinstance(dh[0], Expr):
            self.link_list.append(new_joint)
        self.transform_list.append(new_joint)
        self.sym_theta_list = symarray("theta", len(self.link_list))
    # solve the transformation matrices

    def forward_kinematics(self):
        self.z_jacob = Matrix([])
        transform_to = eye(4)
        for joint in range(0, len(self.transform_list)):
            transform_to = transform_to * self.transform_list[joint].transform()
            self.transform_list[joint].transform_to_me = transform_to
        for joint in range(0, len(self.link_list)):
            self.z_jacob = self.z_jacob.col_insert(1, self.link_list[joint].transform_to_me[0:3, 2]).subs(self.paired_theta)

    def assign_thetas(self, delta_list):
        if len(delta_list) != len(self.link_list):
            print("error")
        else:
            for assign_i in range(0, len(self.link_list)):
                self.link_list[assign_i].update_delta(delta_list[assign_i])

    # create a list of pairs of symbolic thetas and real thetas for substitution
    # match theta_1 to its real number
    def update_theta_list(self):
        self.paired_theta = {}
        self.theta_list = []
        for pair_i in range(0, len(self.link_list)):
            self.theta_list.append(self.link_list[pair_i].theta)
            self.paired_theta[self.sym_theta_list[pair_i]] = self.link_list[pair_i].theta
    # plot the arm in 3d space

    def plot_arm(self):
        x_coords, y_coords, z_coords = [0], [0], [0]

        for link in self.link_list:
            x_coords.append(link.solve_transform(self.paired_theta)[0])
            y_coords.append(link.solve_transform(self.paired_theta)[1])
            z_coords.append(link.solve_transform(self.paired_theta)[2])

        # if goal lies inside the arm high probability of unsolvable singularity
        # TODO move this out of the got dang plot function
        self.check_collisions(x_coords, y_coords, z_coords)

        ax.clear()
        ax.plot(x_coords, y_coords, z_coords)
        ax.plot([self.desiredxyzijk[0]], [self.desiredxyzijk[1]], [self.desiredxyzijk[2]], 'rp')
        x_mod = self.desiredxyzijk[0] + 100 * cos(self.desiredxyzijk[3]) * cos(self.desiredxyzijk[5])
        y_mod = self.desiredxyzijk[1] + 100 * cos(self.desiredxyzijk[4]) * sin(self.desiredxyzijk[3])
        z_mod = self.desiredxyzijk[2] + 100 * sin(self.desiredxyzijk[3])
        ax.plot([self.desiredxyzijk[0], x_mod], [self.desiredxyzijk[1], y_mod], [self.desiredxyzijk[2], z_mod], 'red')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_zlim(0, 700)
        ax.set_ylim(-350, 350)
        ax.set_xlim(0, 700)
        plt.pause(.1)
        plt.draw()

    # run all update functions

    def main_update(self):
        for link in self.link_list:
            link.update_theta()

    def ee_jacobian(self, orientation):
        self.forward_kinematics()
        self.update_theta_list()
        error_with_pose = self.desiredxyzijk - self.transform_list[-1].solve_transform(self.paired_theta)
        error_norm = error_with_pose.norm()
        full_norm = self.desiredxyzijk.norm()

        for i in range(3, 6):
            if error_with_pose[i] > pi:
                error_with_pose[i] = 2*pi - error_with_pose[i]
        if error_with_pose.norm() < 1:
            self.damping_coefficient *= 1.5
        if error_with_pose.norm() < .2:
            return 0
       # self.damping_coefficient = min(1000, int(self.damping_coefficient * full_norm/(error_norm+5)))
        print(error_with_pose)
        print(error_with_pose.norm())
        error_with_pose = np.array(error_with_pose).astype(np.float64)

        transform_matrix = self.transform_list[-1].transform_to_me[0:3, 3]
        jacobian = transform_matrix.jacobian(self.sym_theta_list)
        for jac_i in range(0, 3):
            jacobian = jacobian.row_insert(10, self.z_jacob[jac_i, :])

        jacobian = jacobian.xreplace(self.paired_theta)
        jacobian = np.array(jacobian).astype(np.float64)
        jacobian_transpose = jacobian.transpose()

        jxjt = np.matmul(jacobian, jacobian_transpose)

        i = np.array([[self.damping_coefficient, 0, 0, 0, 0, 0],
                      [0, self.damping_coefficient, 0, 0, 0, 0],
                      [0, 0, self.damping_coefficient, 0, 0, 0],
                      [0, 0, 0, self.damping_coefficient/50, 0, 0],
                      [0, 0, 0, 0, self.damping_coefficient/50, 0],
                      [0, 0, 0, 0, 0, self.damping_coefficient/50]])

        n = np.matmul(np.linalg.inv(jxjt + i), error_with_pose)
        delta_theta = np.matmul(jacobian_transpose, n)

        for update_i in range(0, len(self.link_list)):
            self.link_list[update_i].update_delta(delta_theta[update_i])
        return 1

    def check_collisions(self, x_coords, y_coords, z_coords):
        for coll_i in range(0, len(x_coords) - 1):
            if check_intersect(self.desiredxyzijk[0], self.desiredxyzijk[1], self.desiredxyzijk[2], x_coords[coll_i],
                               y_coords[coll_i], z_coords[coll_i], x_coords[coll_i + 1],
                               y_coords[coll_i + 1], z_coords[coll_i + 1]):
                self.desiredxyzijk += Matrix([0.001, 0.00, -0.001, 0, 0, 0])

    def run_kinematics(self, damping_coefficient, x_coord, y_coord, z_coord, xr, yr, zr):

        self.desiredxyzijk = Matrix([x_coord, y_coord, z_coord, xr, yr, zr])
        self.start_time = time.perf_counter()
        self.damping_coefficient = damping_coefficient
        # zero arm thetas to start position
        for link in self.link_list:
            link.theta = 0

        while arm.ee_jacobian(1) == 1:
            arm.main_update()
            self.desiredxyzijk += Matrix([0, 0, 0, 0, 0, 0])
        arm.plot_arm()


fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
sym_theta_list = symarray("theta", 8)

arm = Armature()

arm.add_link((sym_theta_list[0], -pi / 2, 0, 100))
arm.add_link((sym_theta_list[1], pi / 2, 100, 0))
arm.add_link((sym_theta_list[2], pi / 2, 100, 0))
arm.add_link((sym_theta_list[3], -pi / 2, 100, 0))
arm.add_link((sym_theta_list[4], -pi / 2, 100, 0))
arm.add_link((sym_theta_list[5], pi / 2, 100, 0))
arm.add_link((sym_theta_list[6], 0, 100, 0))
arm.add_link((pi / 2, pi / 2, 0, 0))
arm.add_link((sym_theta_list[7], pi / 2, 0, 0))

arm.main_update()
# arm.plot_arm()
xs = []
ys = []
zs = []

x_rot = pi
y_rot = 0
z_rot = pi / 2
arm.run_kinematics(100, 400, 100, 100, x_rot, y_rot, z_rot)
plt.show()