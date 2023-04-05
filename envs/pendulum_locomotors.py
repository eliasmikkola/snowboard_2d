from envs.robot_bases import XmlBasedRobot, MJCFBasedRobot, URDFBasedRobot
import numpy as np
import pybullet
import os
import pybullet_data
from envs.robot_bases import BodyPart
import pybullet_data
from envs.robot_locomotors import WalkerBase

class PendulumBoard(WalkerBase):
  def __init__(self, bullet_client):
    WalkerBase.__init__(self, "pendulum_board.xml", "board_main", action_dim=2, obs_dim=26, power=0.75)
    #WalkerBase.__init__(self, "snowboard_2d_skis.xml", "torso", action_dim=3, obs_dim=15, power=0.75)
    # paint all parts in green
    
    self._p = bullet_client
  def alive_bonus(self, z, pitch):
    return +1 if z > 0.8 and abs(pitch) < 1.0 else -1
  def robot_specific_reset(self, bullet_client, model_objects):
    self._p = bullet_client
    for j in self.ordered_joints:
      # j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)
      j.reset_current_position(self.np_random.uniform(low=0, high=0), 0)

    
    # left_child_link_index = self.parts["foot_left"].bodyPartIndex
    # right_child_link_index = self.parts["board_right"].bodyPartIndex
    # cid = self._p.createConstraint(model_objects[0], right_child_link_index , model_objects[0], left_child_link_index,self._p.JOINT_FIXED, [0, 0, 0], [-0.4, 0, 0], [0, 0, 0])

    self.body_part_list = ["board_main","front_left", "front_right","back_left", "back_right", "front_capsule", "back_capsule"]
    # board_indices = ["board_right","board_start","board_end"]
    # # changeDynamics for board mass
    # for i in board_indices:
    #   self._p.changeDynamics(model_objects[0], self.parts[i].bodyPartIndex, mass=1.0)
    # for part in self.body_part_list:
    #   print(part, self._p.getDynamicsInfo(model_objects[0], self.parts[part].bodyPartIndex)[0])
    #indices of body parts that are relevant for contact
    self.body_part_indices = [self.parts[f].bodyPartIndex for f in self.body_part_list]

    
    self.contact_points = np.array([0.0 for f in self.body_part_list], dtype=np.float32)
    self.scene.actor_introduce(self)
    self.initial_z = None

  def calc_state(self):
    j = np.array([j.current_relative_position() for j in self.ordered_joints],
                 dtype=np.float32).flatten()
    # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
    self.joint_speeds = j[1::2]
    self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

    body_pose = self.robot_body.pose()
    parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
    self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2]
                    )  # torso z is more informative than mean z
    self.body_real_xyz = body_pose.xyz()
    self.body_orientation = body_pose.orientation()
    self.body_rpy = body_pose.rpy()
    z = self.body_xyz[2]
    if self.initial_z == None:
      self.initial_z = z
    r, p, yaw = self.body_rpy
    self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                        self.walk_target_x - self.body_xyz[0])
    self.walk_target_dist = np.linalg.norm(
        [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
    angle_to_target = self.walk_target_theta - yaw

    rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw),
                                                             np.cos(-yaw), 0], [0, 0, 1]])
    vx, vy, vz = np.dot(rot_speed,
                        self.robot_body.speed())  # rotate speed back to body point of view

    more = np.array(
        [
            z - self.initial_z,
            np.sin(angle_to_target),
            np.cos(angle_to_target),
            0.3 * vx,
            0.3 * vy,
            0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r,
            p
        ],
        dtype=np.float32)

    # print("MORE", more
    contact_points_at_state = self._p.getContactPoints(self.robot_body.bodies[self.robot_body.bodyIndex], -1)
    contact_indices = [contact_point[3] for contact_point in contact_points_at_state]

    for (i, contact) in enumerate(self.body_part_indices):
      if contact in contact_indices:
        self.contact_points[i] = 1.0
      else:
        self.contact_points[i] = 0.0
    # getJointState of all joints
    joint_states = self._p.getJointStates(self.robot_body.bodies[self.robot_body.bodyIndex], self.body_part_indices)
    joint_forces_fx_fz = np.array([np.abs(joint_state[2][0] + joint_state[2][2]) for joint_state in joint_states])
    
    # scale the reaction forces to be between 0 and 1 from 0 to 1000
    joint_reaction_forces = joint_forces_fx_fz / 1000.0
    
    # print("\n\n\n------------------\n------------------\n------------------")
    # for (name, contact) in zip(self.body_part_list, self.contact_points):
    #   print(name, contact)
    # print([more] + [j] + [self.contact_points] + [joint_reaction_forces])
    return np.clip(np.concatenate([more] + [j] + [self.contact_points] + [joint_reaction_forces]), -5, +5)

  def calc_potential(self):
    # the further you go, the more reward you get
    return -self.walk_target_dist / self.scene.dt