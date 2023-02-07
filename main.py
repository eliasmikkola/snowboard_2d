# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gym
import numpy as np
import time
from stable_baselines3 import PPO
import argparse
import wandb


# import mocca_envs
from envs.envs import Walker2DBulletEnv, SnowBoardBulletEnv


def main(args):
    env = SnowBoardBulletEnv(render=args.render)
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
    
    N_TIMESTEPS = args.n_timesteps
    model = PPO("MlpPolicy", env, verbose=1)
    
    
    SAVE_FOLDER = "models"
    
    if args.load:
        # if ppo_snowboard.zip in args.load_model_path
        path_to_load = args.load_model_path
        if "ppo_snowboard.zip" not in args.load_model_path:
            path_to_load = f"{path_to_load}/ppo_snowboard.zip"
        model.load(path_to_load)
    if args.train:
        model.learn(total_timesteps=N_TIMESTEPS)
        # time stamp
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        root_folder = f"{SAVE_FOLDER}/snowboard/{time_stamp}"
        if args.save:
            model.save(f"{root_folder}/ppo_snowboard")
        # hyper parameters
        used_args = vars(args)
        with open(f"{root_folder}/args.txt", "w") as f:
            f.write(str(used_args))

    # TODO: args for training and loading a saved model

    iters = 0
    
    timesteps_per_iter = 2000
    curr_timestep = 0
    while True:
        curr_timestep += 1
        # env.step(np.zeros(6))
        #  step returns state, sum(self.rewards), bool(done), {}
        iters += 1
        actions, _states = model.predict(state)
        if args.dummy:
            actions = np.zeros([13])
        # TODO: add switch for separating model training or zeros for pure env related tweaking/testing
        state, reward, done, _ = env.step(actions)
        sum_reward += reward
        env.render()
        
        x,y,z= env.robot.body_xyz
        # add to x 
        x += 0.5
        z += 0.5
        y += 0.5
        
        env._p.resetDebugVisualizerCamera( cameraDistance=8, cameraYaw=-5, cameraPitch=-40, 
        cameraTargetPosition=[x,y,z])
        if done:
            break
        # if after_done_counter > 200:
        #     print("DONE")
        if args.render:
            time.sleep(0.01)
    
    print("DONE")
    print("sum reward", sum_reward)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-render', type=bool, default=False)
    parser.add_argument('-load', type=bool, default=False)
    parser.add_argument('-load_model_path', type=str, default='')
    parser.add_argument('-train', type=bool, default=False)
    parser.add_argument('-dummy', type=bool, default=False)
    parser.add_argument('-testing', type=bool, default=False)
    parser.add_argument('-save', type=bool, default=True)
    parser.add_argument('-save_folder', type=str, default='models')
    parser.add_argument('-n_timesteps', type=int, default=100000)
    parser.add_argument('-n_episodes', type=int, default=1000)
    parser.add_argument('-use_wandb', type=bool, default=False)


    args = parser.parse_args()
    # load and train are mutually exclusive, print error if both are true
    assert not (args.load and args.train) or (args.load and args.train), "can't train a loaded model"
    assert not (args.load and args.dummy) or (args.load and args.dummy), "can't use -dummy with a loaded model"
    # if load is true, load_model_path should be set
    assert not (args.load and args.load_model_path == '') or (args.load and args.load_model_path != ''), "load_model_path should be set if -load is true"
    main(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
