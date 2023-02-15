from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
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
        self.iteration += 1
        if self.model_args.save_iteration != None and self.iteration % self.model_args.save_iteration == 0:
          self.model.save(f"{self.root_folder}/ppo_snowboard_v{self.iteration}")