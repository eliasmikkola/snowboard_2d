import numpy as np
import pybullet
import gym
from envs.env_bases import MJCFBaseBulletEnv
from envs.robot_locomotors import HumanoidFlagrun, HumanoidFlagrunHarder, HalfCheetah, Walker2D, Hopper, Ant, Humanoid, Snowboard
from envs.scene_stadium import SinglePlayerStadiumScene
from envs.mjcf.utils.generate_plane import SinglePlayerSlopeScene
from matplotlib import colors

class SnowBoardBulletEnv(MJCFBaseBulletEnv):

    def __init__(self, render=False):
        # print("WalkerBase::__init__ start")
        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.stateId = -1
        self._alive = -1
        self.total_steps = 0
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
        self.total_steps = 0
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
        self.robot.robot_body.reset_position([min_x+20, 0, max_z+0])
        # rotate the robot upside down with resetBasePositionAndOrientation
        # self.robot.robot_body.reset_orientation([0, 3.5, 0, 1])
        
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
        self.total_steps += 1
        done = self._isDone()
        STEP_LIMIT = 1500
        if self.total_steps > STEP_LIMIT:
            done = True
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        angs = j[0::2]
        vels = j[1::2]
        kp = 10
        kd = 1
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
                #     print("HEAD CONTACT", contact[9])
                    
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
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)


        electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean(
        ))  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        # TODO: TERMINAL HEAD DAMAGE (if head damage > 1000)
            #print("TERMINAL HEAD DAMAGE")
            # print(self.body_parts_damage[head_index])
        # print(sum(self.body_parts_damage))
        # multiply body part damage by 0.1 to make it less punishing, except head_index which is multiplied by 0.5
        # body_parts_damage_total = self.body_parts_damage
        # for index, damage in enumerate(body_parts_damage_total):
        #     if index == head_index:
        #         body_parts_damage_total[index] = damage * 0.5
        #     else:
        #         body_parts_damage_total[index] = damage * 0.1
        # # sum body part damage
        # print(body_parts_damage_total)
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
        
        ###################
        ##### REWARDS #####
        ###################
        SLOPE_END_BONUS_REWARD = 500
        HEAD_INJURY_BONUD_PENALTY = -1000
        STOPPAGE_PENALTY = -500
        bonus_reward = 0.0
        
        if self.body_parts_damage[head_index] > 1000:
            done = True
            bonus_reward = HEAD_INJURY_BONUD_PENALTY
        # get body parts damage by g(x) = 0.5^x where x is the damage taken
        
        damage_reward = 0.5**np.sum(self.body_parts_damage)

        # body_parts_damage_total = -np.sum(self.body_parts_damage)

        # 1. if 0 damage taken, then reward 1, else exponential decay
        # get robots angle
        # if self.has_touched_ground:

        #### UPRIGHTNESS REWARD ####
        body_y_orientation = np.abs(self.robot.body_orientation[1])
        reward_uprightness = 1.0 - body_y_orientation

        #### VELOCITY REWARD ####
        robot_velocity = self.robot.body_xyz[0] - self.last_xyz[0]

        def calc_reward_velocity(x_difference, max_speed):
          if x_difference <= 0:
              return 0
          else:
              speed = x_difference
              return min(speed / max_speed, 1)
        max_speed = 0.1
        reward_velocity = calc_reward_velocity(robot_velocity, max_speed)
        
        if robot_velocity < 0.0 and self.has_touched_ground:
            self.not_moving_counter += 1
            if self.not_moving_counter > 30:
                
                done = False
                bonus_reward = STOPPAGE_PENALTY
        self.last_xyz = self.robot.body_xyz
        
        # air time reward if contact_points is empty
        air_reward = 1.0 if len(contact_points) == 0 else 0.0
        
        
        #### SLOPE END DETECTION ####
        terrain_plane_pos = self._p.getAABB(self.scene.terrain_plane)
        
        min_x, min_y, min_z = terrain_plane_pos[0]
        max_x, max_y, max_z = terrain_plane_pos[1]
        # if robot x > min_x and robot z < min_z, then slope has ended
        # get robot xyz
        robot_x, robot_y, robot_z = self.robot.body_real_xyz
        if robot_x > min_x and robot_z < min_z:
            print("DONE SLOPE ENDED")
            done = True
            bonus_reward = SLOPE_END_BONUS_REWARD
        
        
        self.rewards = [
            reward_uprightness * 1.0,
            damage_reward * 0.1,
            reward_velocity * 0.1,
            bonus_reward * 1.0,
            air_reward * 0.1
        ]

        debugmode = 0
        if (debugmode):
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)
        
        # reward TODO:, 
        # 1. if 0 damage taken, then reward 1, else exponential decay
        # 2. if robot is moving, then reward 1, else exponential decay
        # 3. if robot is upright, then reward 1, else exponential decay
        # 4. if robot is not touching ground, then reward 1, else exponential decay
        # 5. if robot is not touching ground, and is upside down, then reward 1, else exponential decay

        if bool(done):
            print("DONE, STEPS:", self.total_steps)

        return state, sum(self.rewards), bool(done), {}

      
    def camera_adjust(self):
        x,y,z= self.robot.body_xyz
        # add to x 
        x += 0.5
        z += 0.5
        y += 0.5
        
        self._p.resetDebugVisualizerCamera( cameraDistance=8, cameraYaw=-5, cameraPitch=-40, 
        cameraTargetPosition=[x,y,z])
