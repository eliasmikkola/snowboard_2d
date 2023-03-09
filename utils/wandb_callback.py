from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
# import evaluate_policy
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
import numpy as np
import imageio
import os

class SBCallBack(BaseCallback):

    def __init__(self, root_folder, original_env: VecNormalize, model_args, verbose=0):
        super().__init__(verbose)
        self.steps = 0
        self.original_env = original_env
        self.iteration = 0
        self.root_folder = root_folder
        self.model_args = model_args
        self.ep_rewards = np.array([])
        self.best_mean_reward = -np.inf
        print("original_env", original_env)
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        info = {}
        rewards = self.original_env.normalize_reward(self.original_env.old_reward)
        #normalize reward
        running_mean = self.original_env.ret_rms.mean
        self.steps += 1
        info["steps"] = self.steps
        info["running_mean"] = running_mean
        self.ep_rewards = np.append(self.ep_rewards, rewards)
        # print("IN CALLBACK steps", self.steps, "running_mean", running_mean, "rewards", rewards)
        if self.steps % 1000 == 0:
          if self.model_args.use_wandb:
            # print("IN CALLBACK steps", self.steps, "running_mean", running_mean, "rewards", rewards)
            ep_mean = np.mean(self.ep_rewards)
            info["ep_mean"] = ep_mean
            self.ep_rewards = []
            wandb.log(info)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass
    
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass 
    def save_video_on_training(self, folder):
        rgb_frames = np.array([])
        state = self.original_env.reset()
        # set env to eval mode
        self.original_env.training = False
        steps_video = 0
        while True:
            # env.step(np.zeros(6))
            #  step returns state, sum(self.rewards), bool(done), {}
            steps_video += 1
            print("steps_video", steps_video)
            actions, _states = self.model.predict(state, deterministic=True)
            # print("actions", actions)
            state, reward, done, _ = self.original_env.step(actions)
            
            if steps_video % 3 == 0:
                rgb_arr = self.original_env.render(mode='rgb_array')
                rbg_arr_rows = int(len(rgb_arr)) # as int
                rbg_arr_cols = int(len(rgb_arr[0])) # as int
                rgb_arr_sliced = rgb_arr[0:rbg_arr_rows, 0:rbg_arr_cols]
                rgb_frames = np.append(rgb_frames, rgb_arr_sliced)
            # if one in done is true, then episode is done
            if done.any():
                break
            # path_to_load but replace models with videos and 

        imageio.mimsave(f"{folder}/video.gif", rgb_frames, fps=20)
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.original_env.training = False
        iteration_folder = f"{self.root_folder}/iter_{self.iteration}"
        print("rollout start", self.iteration)
        if self.model_args.save_iteration != None and self.iteration % self.model_args.save_iteration == 0:
            self.model.save(f"{iteration_folder}/model")
            # save policy weights
            stats_path = f"{iteration_folder}/stats.pth"
            # save stats for normalization
            self.original_env.save(stats_path)
            # save text file with both paths
            with open(f"{iteration_folder}/args.txt", "w") as f:
                f.write(f"python3 main.py --load --no_save --user_input  --model_path={iteration_folder} --save_video")
        if self.iteration % self.model_args.eval_period == 0:
            # Evaluate the trained agent
            mean_reward, std_reward = evaluate_policy(self.model, self.original_env, n_eval_episodes=20, deterministic=True)
            print(f"eval_mean_reward={mean_reward:.2f} +/- {std_reward}")
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                int_mean_reward = int(mean_reward)
                self.model.save(f"{iteration_folder}/evaluated/model")
                # save policy weights, add reward as int to name
                stats_path = f"{iteration_folder}/evaluated/stats.pth"
                # save stats for normalization
                self.original_env.save(stats_path)
                # save text file with both paths "python3 main.py --load --no_save --user_input  --model_path={path} --save_video"
                with open(f"{iteration_folder}/best_args_{int_mean_reward}.txt", "w") as f:
                    f.write(f"python3 main.py --load --no_save --user_input  --model_path={iteration_folder}/evaluated --save_video")
            if self.model_args.use_wandb:
                wandb.log({"eval_mean_reward": mean_reward, "std_reward": std_reward, "ppo_iteration": self.iteration, "steps": self.steps})
            # if self.model_args.save_video:
            #     self.save_video_on_training(folder=iteration_folder)
        self.iteration += 1
        self.original_env.training = True
