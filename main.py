# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gym
import numpy as np
import time
from stable_baselines3 import A2C


# import mocca_envs
from envs.envs import Walker2DBulletEnv, SnowBoardBulletEnv


def main():
    env = SnowBoardBulletEnv(render=True)
    state = env.reset()
    # print observation space and action space
    print("OBS space", env.observation_space.shape)
    print("ACT space", env.action_space.shape)
    print("state", state.shape)


    action_pieces = np.random.random([6, 6]) * 0
    # keypoints
    action_pieces[0] = np.array([0, 0, 0, 0, 0, 0])  # first keypoint
    action_pieces[1] = np.array([-0.5, 0, 0, 0, 0, 0])
    action_pieces[2] = np.array([0, 0, 0, 0, 0, 0])
    action_pieces[3] = np.array([-1, 0, 0, 0, 0, 0])
    action_pieces[4] = np.array([0, 0, 0, 0, 0, 0])
    action_pieces[5] = np.array([0, 0, 0, 0, 0, 0])

    # after interpolation
    actions = []
    actions.append(np.linspace(action_pieces[0], action_pieces[1], 50))
    actions.append(np.linspace(action_pieces[1], action_pieces[2], 50))
    actions.append(np.linspace(action_pieces[2], action_pieces[3], 50))
    actions.append(np.linspace(action_pieces[3], action_pieces[4], 50))
    actions.append(np.linspace(action_pieces[4], action_pieces[5], 50))
    actions = np.concatenate(actions, axis=0)

    # actions[:, 0] = 1
    # actions[:, 1] = 0
    # actions[:, 3] = 0
    # actions[:, 4] = 0

    # TODO: find action_pieces such that sum_reward is large. Use something like CEM or CMA-ES.

    sum_reward = 0
    state = env.reset()
    after_done_counter = 0
    
    N_TIMESTEPS = 10000
    model = A2C('MlpPolicy', env, verbose=1)
    
    
    SAVE_FOLDER = "a2c_snowboard"

    # model.load(f"{SAVE_FOLDER}/a2c_snowboard")
    model.learn(total_timesteps=N_TIMESTEPS)
    model.save(f"{SAVE_FOLDER}/a2c_snowboard")
    # TODO: args for training and loading a saved model

    iters = 0
    while True:
        # env.step(np.zeros(6))
        #  step returns state, sum(self.rewards), bool(done), {}
        iters += 1
        actions, _states = model.predict(state)
        # TODO: add switch for separating model training or zeros for pure env related tweaking/testing
        # actions = np.zeros([13])
        state, reward, done, _ = env.step(actions)
        sum_reward += reward
        env.render()
        
        x,y,z= env.robot.body_xyz
        # add to x 
        x += 0.5
        z += 0.5
        y += 0.5
        
        if iters % 100 == 0:
            print("step",iters, "reward", sum_reward)
        env._p.resetDebugVisualizerCamera( cameraDistance=8, cameraYaw=-5, cameraPitch=-40, 
        cameraTargetPosition=[x,y,z])
        if done:
            after_done_counter += 1
            break
        # if after_done_counter > 200:
        #     print("DONE")
        # time.sleep(0.01)
    print("DONE")

    # import pybullet as p
    # import time
    # import pybullet_data
    # physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    # p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    # p.setGravity(0, 0, -10)
    # planeId = p.loadURDF("plane.urdf")
    # startPos = [0, 0, 1]
    # startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    # boxId = p.loadURDF("r2d2.urdf", startPos, startOrientation)
    # # set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
    # for i in range(10000):
    #     p.stepSimulation()
    #     time.sleep(1. / 240.)
    # cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    # print(cubePos, cubeOrn)
    # p.disconnect()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
