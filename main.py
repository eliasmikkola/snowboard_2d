# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gym
import numpy as np
import time

# import mocca_envs
from envs.envs import Walker2DBulletEnv, SnowBoardBulletEnv


def main():
    env = SnowBoardBulletEnv(render=True)
    env.reset()

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
    env.reset()
    for a in actions:
        # env.step(np.zeros(6))
        #_, reward, _, _ = env.step(a)
        _, reward, _, _ = env.step(np.zeros(6))
        sum_reward += reward
        env.render()
        time.sleep(0.066)
    print(sum_reward)

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
