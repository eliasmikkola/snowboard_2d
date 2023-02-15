# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gym
import numpy as np
import time
from stable_baselines3 import PPO
import argparse
import wandb
import keyboard
import pybullet as p
from utils.wandb_callback import SBCallBack
# import mocca_envs
from envs.snowboard_env import SnowBoardBulletEnv
import imageio
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
import os
# gym.register('SnowBoarding-v0', entry_point=SnowBoardBulletEnv)

# def make_single_custom_env():
#     return gym.make("SnowBoarding-v0")
# is this the correct way?

def main(args):
     # Create a function to handle the key events
    action_direction = 0
    def on_press_event(e):
        nonlocal action_direction
        print("key pressed")
        key = e.name
        # Update the actions of the RL model based on the pressed key
        if key == 'up':
            # Move the robot forward
            action_direction = 0
            print("up")
        elif key == 'down':
            # Move the robot backward
            pass
        elif key == 'left':
            # Turn the robot to the left
            action_direction = -1
            print("left")
        elif key == 'right':
            # Turn the robot to the right
            action_direction = 1
            print("rght")
    
    # Register the key event listener
    keyboard.hook(on_press_event)
    wandb_run = None
    if args.use_wandb:
        if args.wandb_resume:
            # resume from previous run
            wandb_run = wandb.init(project="Snowboard_2d", allow_val_change=True, resume="must", id=args.wandb_resume)
            wandb_run.config.update({"allow_val_change": True})
        else:
            wandb_run = wandb.init(project="Snowboard_2d")
        if args.run_name is None:
            args.run_name = wandb.run.id
        # wandb_run.config.update(args)
        wandb_run.config.update({"allow_val_change": True})
    
    SAVE_FOLDER = "models"
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    ROOT_FOLDER = f"{SAVE_FOLDER}/snowboard/{time_stamp}-{args.run_name}"
    
    run_id = args.run_name

    def create_env():
        mode = 'rgb_array' if args.save_video else 'human'
        return SnowBoardBulletEnv(render=args.render, wandb_instance=wandb_run, render_mode=mode)
    num_envs = 8
    multi_env = False
    if args.train or args.retrain:
        print("Creating SubprocVecEnv ENV")
        env = SubprocVecEnv([create_env for i in range(num_envs)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=np.inf, clip_reward=np.inf)
        print("VEC ENC", env, env.old_reward)

        multi_env = True
    else:
        print("Creating single snowboard env")
        env = create_env()
        multi_env = False


    state = env.reset()
    # print observation space and action space
    print("OBS space", env.observation_space.shape)
    print("ACT space", env.action_space.shape)
    print("state", state.shape)

    state = env.reset()
    after_done_counter = 0
    
    N_TIMESTEPS = args.timesteps
    model = PPO("MlpPolicy", env, verbose=1)
    
    
    def save_model(model):
        root_folder = ROOT_FOLDER
        if args.retrain:
            root_folder = f"{root_folder}_retrained"
        # if args.retrain:
        #     root_folder = f"{root_folder}_retrained({args.model})"

        save_path = f"{root_folder}/ppo_snowboard"
        model.save(save_path)
        # hyper parameters
        used_args = vars(args)
        # open or create file
        with open(f"{root_folder}/args.txt", "w") as f:
            f.write(str(used_args))
    path_to_load = None
    if args.load:
        # if ppo_snowboard.zip in args.model
        path_to_load = args.model
        if ".zip" not in args.model:
            path_to_load = f"{path_to_load}/ppo_snowboard.zip"
        model.load(path_to_load)
    if args.retrain:
        path_to_load = args.model
        model = PPO.load(path_to_load)
        model.set_env(env)
        # continue from previous training
        model.learn(total_timesteps=N_TIMESTEPS, progress_bar=True, reset_num_timesteps=False , callback=SBCallBack(root_folder=ROOT_FOLDER, original_env=env, model_args=args))
        # model.learn(total_timesteps=N_TIMESTEPS ,callback=None, seed=None,
        #     log_interval=1, tb_log_name="Logs", reset_num_timesteps=False)
    if args.train or args.retrain:
        model.learn(total_timesteps=N_TIMESTEPS,  progress_bar=True, callback=SBCallBack(root_folder=ROOT_FOLDER, original_env=env, model_args=args))
        if not args.no_save and (args.train or args.retrain):
            save_model(model)


    iters = 0
    
    iterations = 100
    total_rewards = 0
    action = 0
    if not args.train and not args.retrain:
        for i in range(iterations):
            rgb_frames = []
            print ("ITERATION", i)
            state = env.reset()
            curr_timestep = 0
            actions_all = np.zeros([13])
            sum_reward = 0
            while True:
                # try: 
                    # Get the current keyboard state
                    if args.user_input:
                        keyboard_events = p.getKeyboardEvents()
                        # Update the action variable based on the pressed key
                        if 65297 in keyboard_events:
                            action = 0
                        elif 65298 in keyboard_events:
                            action = 0
                        elif 65295 in keyboard_events:
                            action = -1
                        elif 65296 in keyboard_events:
                            action = 1
                    
                    curr_timestep += 1
                    # env.step(np.zeros(6))
                    #  step returns state, sum(self.rewards), bool(done), {}
                    iters += 1

                    actions, _states = model.predict(state, deterministic=True)
                    # print("actions", actions)
                    actions_all += actions
                    if args.dummy:

                        #actions = np.random.uniform(-1, 1, size=13)
                        # fill np array with action_direction
                        actions = np.ones([13]) * action
                    # TODO: add switch for separating model training or zeros for pure env related tweaking/testing
                    state, reward, done, _ = env.step(actions)
                    sum_reward += reward
                    if not multi_env:
                        if args.render:
                            env.render(mode='human')
                        elif args.save_video:
                            rgb_arr = env.render(mode='rgb_array')
                            rgb_frames.append(rgb_arr)

                    if not multi_env and done:
                        break
                    elif multi_env and done.all():
                        break
                    # if after_done_counter > 200:
                    #     print("DONE")
                    if args.render:
                        time.sleep(0.01)
                # except KeyboardInterrupt:
                #     if not args.no_save:
                #         print("Saving model in interrupt")
                #         save_model(model)
                #     break

            print("TIME STEPS:", curr_timestep)
            print("sum reward", sum_reward)
            # create gif from rgb_frames
            if args.save_video:
                # path_to_load but replace models with videos and 
                path_to_save_video = path_to_load.replace("models", "videos")
                if not os.path.exists(path_to_save_video):
                    os.makedirs(path_to_save_video)
                # speed up video
                rgb_frames = rgb_frames[::2]
                # imageio 60 fps
                imageio.mimsave(f"{path_to_save_video}/video_{i}.gif", rgb_frames, fps=60)
        # print("actions", actions_all)
        total_rewards += sum_reward
    print("total rewards mean",  total_rewards/iterations, f"({total_rewards})")
    wandb_run.log({"ep_reward": sum_reward})
    

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render',  action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--model', type=str, default='')
    # parser.add_argument('--train', action='store_true')
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--save_folder', type=str, default='models')
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--save_iteration', type=int, default=1)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--wandb_resume', type=str)
    parser.add_argument('--user_input', action='store_true')
    parser.add_argument('--save_video', action='store_true')
    

    args = parser.parse_args()
    print(args)
    # load and train are mutually exclusive, print error if both are true
    # assert (args.load and not args.train) or (not args.load and args.train), "can't train a loaded model"
    assert not (args.load and args.dummy) or (args.load and args.dummy), "can't use -dummy with a loaded model"
    # if load is true, model should be set
    assert not (args.load and args.model == '') or (args.load and args.model != ''), "model should be set if -load is true"
    # assert model to contain .zip
    assert not (args.load and ".zip" not in args.model) or (args.load and ".zip" in args.model), "model should contain .zip"
    main(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
