from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
# import evaluate_policy
from stable_baselines3.common.evaluation import evaluate_policy
# import Monitor
from stable_baselines3.common.monitor import Monitor
# import PPO
from stable_baselines3 import PPO
from envs.snowboard_env import SnowBoardBulletEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

import wandb
import numpy as np
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
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        print("rollout start", self.iteration)
        if self.model_args.save_iteration != None and self.iteration % self.model_args.save_iteration == 0:
            self.model.save(f"{self.root_folder}/ppo_snowboard_v{self.iteration}")
            # save policy weights
            stats_path = f"{self.root_folder}/stats_v{self.iteration}.pth"
            # save stats for normalization
            self.original_env.save(stats_path)
        if self.iteration % self.model_args.eval_period == 0:
            
            self.original_env.training = False
            mean_reward, std_reward = evaluate_policy(self.model, self.original_env, n_eval_episodes=20, deterministic=True)
            

            print(f"eval_mean_reward={mean_reward:.2f} +/- {std_reward}")
            
            self.best_mean_reward = mean_reward
            # Example for saving best model
            print("Saving new best model")
            int_mean_reward = int(mean_reward)
            save_model_path = f"{self.root_folder}/best_model_{self.iteration}_reward_{int_mean_reward}"
            self.model.save(save_model_path)
            # save policy weights, add reward as int to name
            stats_path = f"{self.root_folder}/best_stats_{self.iteration}_reward_{int_mean_reward}.pth"
            # save stats for normalization
            self.original_env.save(stats_path)
            print("mean reward", mean_reward, "std reward", std_reward)

            # load saved model
            # fresh env like env with saved stats
            def create_env():
                return SnowBoardBulletEnv(render=False, wandb_instance=None, render_mode="human")
            env_new = SubprocVecEnv([create_env for i in range(8)])
            
            env_new = Monitor(env_new)
            env_new = VecNormalize(env_new, norm_obs=True, norm_reward=False, clip_obs=np.inf, clip_reward=np.inf)
            env_new.training = False
            model_new = PPO.load(save_model_path, env=env_new)
            model_new.set_env(env_new)
            mean_reward_new, std_reward_new = evaluate_policy(model_new, env_new, n_eval_episodes=20, deterministic=True)
            print("mean reward with loaded model and env", mean_reward_new, "std reward new", std_reward_new)
            self.original_env.training = True

            if self.model_args.use_wandb:
                wandb.log({"eval_mean_reward": mean_reward, "std_reward": std_reward, "ppo_iteration": self.iteration, "steps": self.steps})
        self.iteration += 1