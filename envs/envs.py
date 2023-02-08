import numpy as np
import pybullet
import gym
from envs.env_bases import MJCFBaseBulletEnv
from envs.robot_locomotors import HumanoidFlagrun, HumanoidFlagrunHarder, HalfCheetah, Walker2D, Hopper, Ant, Humanoid, Snowboard
from envs.scene_stadium import SinglePlayerStadiumScene
from envs.mjcf.utils.generate_plane import SinglePlayerSlopeScene
from matplotlib import colors

class WalkerBaseBulletEnv(MJCFBaseBulletEnv):

    def __init__(self, robot, render=False):
        # print("WalkerBase::__init__ start")
        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.stateId = -1
        MJCFBaseBulletEnv.__init__(self, robot, render)

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                      gravity=9.8,
                                                      # gravity=0,
                                                      timestep=0.0165 / 4,
                                                      frame_skip=4)
        return self.stadium_scene

    def reset(self):
        if (self.stateId >= 0):
            # print("restoreState self.stateId:",self.stateId)
            self._p.restoreState(self.stateId)

        r = MJCFBaseBulletEnv.reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        # color all body parts to red
        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
            self._p, self.stadium_scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex],
                                self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        if (self.stateId < 0):
            self.stateId = self._p.saveState()
            # print("saving state self.stateId:",self.stateId)
        return r

    def _isDone(self):
        return self.did_fall

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(
            init_x, init_y, init_z
        )  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)

    electricity_cost = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.1  # discourage stuck joints

    def step(self, a):
        j = np.array([j.current_position() for j in self.ordered_joints],
                     dtype=np.float32).flatten()
        lo = np.array([j.lowerLimit for j in self.ordered_joints],
                     dtype=np.float32).flatten()
        hi = np.array([j.upperLimit for j in self.ordered_joints],
                     dtype=np.float32).flatten()

        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        angs = j[0::2]
        vels = j[1::2]
        kp = 1e1
        kd = 1e1
        target_angs = a * np.pi
        # target_angs = a * (hi - lo) / 2 + (hi + lo) / 2
        # target_angs = np.zeros_like(a)
        # target_angs[a > 0] = hi[a > 0] * a[a > 0]
        # target_angs[a < 0] = -lo[a < 0] * a[a < 0]
        torque = a # kp * (target_angs - angs) + kd * (0 - vels)
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(torque)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        self._alive = float(
            self.robot.alive_bonus(
                state[0] + self.robot.initial_z,
                self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        done = self._isDone()
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet
        ):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean(
        ))  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(self._alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)
        
        # get robot's velocity
        robot_velocity = self.robot.body_xyz[0] - self.robot.prev_xpos
        print("robot_velocity", robot_velocity)

        self.rewards = [
            self._alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost
        ]
        if (debugmode):
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}

    def camera_adjust(self):
        x, y, z = self.robot.body_xyz
        self.camera_x = x
        self.camera.move_and_look_at(x, y, z, x, y, z)

class SnowBoardBulletEnv(MJCFBaseBulletEnv):

    def __init__(self, render=False):
        # print("WalkerBase::__init__ start")
        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.stateId = -1
        self._alive = -1
        self.did_fall = False
        self.last_xyz = [0, 0, 0]
        self.has_touched_ground = False
        self.not_moving_counter = 0
        self.robot = Snowboard(bullet_client=self)
        MJCFBaseBulletEnv.__init__(self, self.robot, render)
        
        self.reset()
        self.body_parts_damage = np.zeros(len(self.robot.parts))


    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = SinglePlayerSlopeScene(bullet_client,
                                                      gravity=9.8,
                                                      # gravity=0,
                                                      timestep=0.0165 / 4,
                                                      frame_skip=4)
        return self.stadium_scene

    def reset(self):
        if (self.stateId >= 0):
            # print("restoreState self.stateId:",self.stateId)
            self._p.restoreState(self.stateId)

        r = MJCFBaseBulletEnv.reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
            self._p, self.stadium_scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex],
                                self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        if (self.stateId < 0):
            self.stateId = self._p.saveState()
            # print("saving state self.stateId:",self.stateId)
        
        # set robot friction
        # board indices 
        board_indices = [self.robot.parts["board_right"].bodyPartIndex , self.robot.parts["board_start"].bodyPartIndex, self.robot.parts["board_end"].bodyPartIndex]
        # map body parts to the body index not in board_indices
        body_indices = [self.robot.parts[part].bodyPartIndex for part in self.robot.parts if part not in ["board_right", "board_start", "board_end"]]
        for i in body_indices:
            self._p.changeDynamics(self.robot.robot_body.bodies[self.robot.robot_body.bodyIndex], i, lateralFriction=0.7, spinningFriction=0.7, rollingFriction=0.7)
            

        # get the terrain plane box position
        terrain_plane_pos = self._p.getAABB(self.scene.terrain_plane)
    
        min_x, min_y, min_z = terrain_plane_pos[0]
        max_x, max_y, max_z = terrain_plane_pos[1]

        # spawn the robot at the top of the terrain plane
        self.robot.robot_body.reset_position([min_x+3, 0, max_z])
        
        # camera look at the robot
        self.camera_adjust()
        
        # color the robot
        for body_part_key in self.robot.parts:
            body_part = self.robot.parts[body_part_key]
            
            body_part_index = body_part.bodyPartIndex
            self._p.changeVisualShape(self.robot.robot_body.bodies[self.robot.robot_body.bodyIndex], body_part_index, rgbaColor=[0.7, 0.7, 0.7, 1])
        return r

    def _isDone(self):
        return self.did_fall
    def touches_ground(self):
        contact_points = self._p.getContactPoints(self.robot.robot_body.bodies[self.robot.robot_body.bodyIndex], -1)

        # if "board_right" "board_start" "board_end" in contact_points:
        board_indices = [self.robot.parts["board_right"].bodyPartIndex , self.robot.parts["board_start"].bodyPartIndex, self.robot.parts["board_end"].bodyPartIndex]

        head_index = self.robot.parts["head"].bodyPartIndex
        #  enumerate contact points
        for index, contact in enumerate(contact_points):
            # print body name
            if contact[3] not in board_indices:
                if contact[3] == head_index:
                    if contact[9] > 30.0:
                        print("foo")
                #self.did_fall = True
                return True
        return False
    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(
            init_x, init_y, init_z
        )  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)

    electricity_cost = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.1  # discourage stuck joints

    def step(self, a):
        j = np.array([j.current_position() for j in self.ordered_joints],
                     dtype=np.float32).flatten()
        lo = np.array([j.lowerLimit for j in self.ordered_joints],
                     dtype=np.float32).flatten()
        hi = np.array([j.upperLimit for j in self.ordered_joints],
                     dtype=np.float32).flatten()

        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        angs = j[0::2]
        vels = j[1::2]
        kp = 1e1
        kd = 1e2
        target_angs = a * np.pi
        # target_angs = a * (hi - lo) / 2 + (hi + lo) / 2
        # target_angs = np.zeros_like(a)
        # target_angs[a > 0] = hi[a > 0] * a[a > 0]
        # target_angs[a < 0] = -lo[a < 0] * a[a < 0]
        # touches_ground = self.touches_ground()
        contact_points = self._p.getContactPoints(self.robot.robot_body.bodies[self.robot.robot_body.bodyIndex], -1)
        # if contact_points not ()
        if contact_points:
            self.has_touched_ground = True
        # if "board_right" "board_start" "board_end" in contact_points:
        board_indices = [self.robot.parts["board_right"].bodyPartIndex , self.robot.parts["board_start"].bodyPartIndex, self.robot.parts["board_end"].bodyPartIndex]

        head_index = self.robot.parts["head"].bodyPartIndex
        def color_from_value(value):
            cmap = colors.LinearSegmentedColormap.from_list("",["yellow","orange","red","black"])
            if value < 0.0 or value > 1.0:
                raise ValueError("Value must be between 0.0 and 1.0")
            rgba = cmap(value)
            return rgba
        #  enumerate contact points
        for index, contact in enumerate(contact_points):
            # print body name
            if contact[3] not in board_indices:
                # print("PLANE", self.scene.terrain_plane)
                self._p.changeDynamics(self.scene.terrain_plane, -1, lateralFriction=1, spinningFriction=1, rollingFriction=1)
                #print("contact", contact[3])

                # from self.robot.parts.values() find the body part that has the same bodyPartIndex as contact[3]
                # body_part = next((body_part for body_part in self.robot.parts.values() if body_part.bodyPartIndex == contact[3]), None)

                # if contact[3] == head_index:
                #     #self.did_fall = True
                #     print("HEAD CONTACT")
                #     if contact[9] > 30.0:
                #         print("foo")
                contact_index = contact[3]
                damage = contact[9]

                if contact[9] > 30.0:
                    self.body_parts_damage[contact_index] += damage
                    # print(contact_index, self.body_parts_damage[contact_index])
                    # create rgb color gradient [r,g,b,a] from light yellow to dark red
                    damage_taken = self.body_parts_damage[contact_index]
                    fatal_damage = 40000.0
                    damage_portion = min(damage_taken, fatal_damage)
                    ratio = damage_portion / fatal_damage
                    rgb_array = color_from_value(ratio)

                    
                    damage_overkill = damage_taken - 60000
                    if damage_overkill > 0:
                        rgb_array = [0, 0, 0, 1]

                    self._p.changeVisualShape(self.robot.robot_body.bodies[self.robot.robot_body.bodyIndex], contact_index, rgbaColor=rgb_array)
            else: 
                self._p.changeDynamics(self.scene.terrain_plane, -1, lateralFriction=0.0, spinningFriction=0.01, rollingFriction=0.01)
                
        torque = kp * (target_angs - angs) + kd * (0 - vels)
        # if not self.did_fall:
        # else:
        #     torque = a
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(torque)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit
        
        # alive if body height above "terrain" get contact with ground
        # get body indices of contact points
        
        # if other body indices in contact points, then not alive
        


        float(
            self.robot.alive_bonus(
                state[0] + self.robot.initial_z,
                self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        done = self._isDone()
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        # feet_collision_cost = 0.0
        # for i, f in enumerate(
        #         self.robot.feet
        # ):  # TODO: Maybe calculating feet contacts could be done within the robot code
        #     contact_ids = set((x[2], x[4]) for x in f.contact_list())
        #     # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
        #     if (self.ground_ids & contact_ids):
        #         # see Issue 63: https://github.com/openai/roboschool/issues/63
        #         # feet_collision_cost += self.foot_collision_cost
        #         self.robot.feet_contact[i] = 1.0
        #     else:
        #         self.robot.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean(
        ))  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0

        # multiply body part damage by 0.1 to make it less punishing, except head_index which is multiplied by 0.5
        body_parts_damage_total = self.body_parts_damage
        for index, damage in enumerate(body_parts_damage_total):
            if index == head_index:
                body_parts_damage_total[index] = damage * 0.5
            else:
                body_parts_damage_total[index] = damage * 0.1
        # sum body part damage
        body_parts_damage_total = -np.sum(body_parts_damage_total)

        if (debugmode):
            print("alive=")
            print(self._alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("body_parts_damage")
            print(self.body_parts_damage)
        
        # get robot's velocity
        robot_velocity = self.robot.body_xyz[0] - self.last_xyz[0]
        # print("robot_velocity", robot_velocity)
        if robot_velocity < 0.0 and self.has_touched_ground:
            self.not_moving_counter += 1
            if self.not_moving_counter > 30:
                # print("ROBOT STOPPED - DONE")
                done = True
        self.last_xyz = self.robot.body_xyz

        self.rewards = [
            self._alive, progress, electricity_cost, joints_at_limit_cost, body_parts_damage_total
        ]
        if (debugmode):
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)
        


        terrain_plane_pos = self._p.getAABB(self.scene.terrain_plane)
        # print("pos", terrain_plane_pos)
        min_x, min_y, min_z = terrain_plane_pos[0]
        max_x, max_y, max_z = terrain_plane_pos[1]
        # if robot x > min_x and robot z < min_z, then slope has ended
        # get robot xyz
        robot_x, robot_y, robot_z = self.robot.body_real_xyz
        if robot_x > min_x and robot_z < min_z:
            print("DONE SLOPE ENDED")
            done = True

        return state, sum(self.rewards), bool(done), {}

    def camera_adjust(self):
        x,y,z= self.robot.body_xyz
        # add to x 
        x += 0.5
        z += 0.5
        y += 0.5
        
        self._p.resetDebugVisualizerCamera( cameraDistance=8, cameraYaw=-5, cameraPitch=-40, 
        cameraTargetPosition=[x,y,z])


class HopperBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, render=False):
        self.robot = Hopper()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)


class Walker2DBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, render=False):
        self.robot = Walker2D()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)


class HalfCheetahBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, render=False):
        self.robot = HalfCheetah()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)

    def _isDone(self):
        return False


class AntBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, render=False):
        self.robot = Ant()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)


class HumanoidBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, robot=None, render=False):
        if robot is None:
            self.robot = Humanoid()
        else:
            self.robot = robot
        WalkerBaseBulletEnv.__init__(self, self.robot, render)
        self.electricity_cost = 4.25 * WalkerBaseBulletEnv.electricity_cost
        self.stall_torque_cost = 4.25 * WalkerBaseBulletEnv.stall_torque_cost


class HumanoidFlagrunBulletEnv(HumanoidBulletEnv):
    random_yaw = True

    def __init__(self, render=False):
        self.robot = HumanoidFlagrun()
        HumanoidBulletEnv.__init__(self, self.robot, render)

    def create_single_player_scene(self, bullet_client):
        s = HumanoidBulletEnv.create_single_player_scene(self, bullet_client)
        s.zero_at_running_strip_start_line = False
        return s


class HumanoidFlagrunHarderBulletEnv(HumanoidBulletEnv):
    random_lean = True  # can fall on start

    def __init__(self, render=False):
        self.robot = HumanoidFlagrunHarder()
        self.electricity_cost /= 4  # don't care that much about electricity, just stand up!
        HumanoidBulletEnv.__init__(self, self.robot, render)

    def create_single_player_scene(self, bullet_client):
        s = HumanoidBulletEnv.create_single_player_scene(self, bullet_client)
        s.zero_at_running_strip_start_line = False
        return s
