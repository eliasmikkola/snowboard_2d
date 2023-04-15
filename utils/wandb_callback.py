from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
# import evaluate_policy
from stable_baselines3.common.evaluation import evaluate_policy
# import ActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
import wandb
import numpy as np
import imageio
import os
import torch as th
def create_retrain_script(root_folder, iteration_folder, model_args, slope_params):
    # this file is in utils, cd .. to root
    # set the path to the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # set the path to the parent directory of the model directory
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    # write file to iteration_folder/job_continue.sh
    shell_script_path = os.path.join(iteration_folder, 'job_continue.sh')
    
    model_dir = os.path.join(root_folder, 'models')


    # get the absolute whole path of iteration_folder and root_folder
    iteration_folder = os.path.abspath(iteration_folder)
    root_folder = os.path.abspath(root_folder)
    
    # read from dict slope_params

    # generate the shell script
    with open(shell_script_path, 'w') as f:
        f.write(f'''#!/bin/sh
#!/bin/sh
# init and activate conda environment
source ~/.bashrc
conda init
conda activate snow

# add model path to variable
MODEL_PATH={iteration_folder}
WANDB_RESUME={model_args.run_name}

if [ "$SLURM_JOB_NAME" = "bash" ]; then
    echo "SLURM_JOB_NAME is not set"
    exit 1
fi

# this script is calle from root, cd to RUNS/cold-start-flip/
cd {root_folder}

# cd to the location folder from where 
# cp -r $MODEL_PATH MODEL_START_9
python3 main.py --retrain --ppo_steps={model_args.ppo_steps} --timesteps={model_args.timesteps} --save_iteration={model_args.save_iteration} --eval_period={model_args.eval_period} --use_wandb --model_path=MODEL_START --wandb_resume=$WANDB_RESUME --S={slope_params[0]} --A={slope_params[1]} --F={slope_params[2]} --reward_threshold={model_args.reward_threshold}''')

    # create evaluation script 
    eval_script_path = os.path.join(iteration_folder, 'job_eval.sh')
    
    with open(eval_script_path, 'w') as f:
        f.write(f'''#!/bin/sh
# init and activate conda environment
source ~/.bashrc
conda init
conda activate snow

# cd to the location folder from where 
cd {parent_dir}/
MODEL_PATH={iteration_folder}
python3 main.py --load --no_save --model_path=$MODEL_PATH --save_video --S={slope_params[0]} --A={slope_params[1]} --F={slope_params[2]}''')



class SBCallBack(BaseCallback):

    def __init__(self, root_folder, original_env: VecNormalize, model_args, slope_feasibility_arr, verbose=0):
        super().__init__(verbose)
        self.steps = 0
        self.original_env = original_env
        self.slope_feasibility_arr =slope_feasibility_arr
        self.iteration = 0
        self.eval_iteration = 0
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
        iteration_folder = f"{self.root_folder}/runs/iter_{self.iteration}"
        
        # get values from value offrom PPO policy.forward for each slope
        slope_distribution = np.zeros(len(self.slope_feasibility_arr))
        for index, (i_steep, i_amp, i_freq, i_is_feasible) in enumerate(self.slope_feasibility_arr):
            # set i to 1 and all others to 0
            if i_is_feasible == 0:
                slope_distribution[index] = 0
            else:
                distribution_to_set = np.zeros(len(self.slope_feasibility_arr))
                # set index to 1
                distribution_to_set[index] = 1
                obs_state = self.original_env.venv.env_method("parameterized_reset", [distribution_to_set, self.slope_feasibility_arr], indices=0)
                # create a torch tensor from the obs_state
                obs_tensor = th.tensor(obs_state[0])
                obs_tensor = obs_tensor.reshape(1,68)
                
                policy: ActorCriticPolicy = self.model.policy
                net_values = policy.forward(obs_tensor)
                # form normalized probabilities from values, set to i of distribution
                slope_distribution[index] = net_values[1].item()
        # set values proportional to the probability of the slope being feasible with exp(-10x)
        feas = self.slope_feasibility_arr[:,3]
        dist = slope_distribution
        p = np.exp(-dist)
        p = p*feas
        p /= p.sum()
        normalized_dist = p
        

        self.original_env.venv.env_method("parameterized_reset", [normalized_dist, self.slope_feasibility_arr])
        # optimize the above with list comprehension
        # set slope params to max

        # slope_distribution_comprehensive = np.array([self.model.policy.forward(obs_state)[1][0].item() if i_is_feasible else 0 for i_steep, i_amp, i_freq, i_is_feasible in self.slope_feasibility_arr])

        print("rollout start", self.iteration)
        if self.model_args.save_iteration != None and self.iteration % self.model_args.save_iteration == 0:
            self.model.save(f"{iteration_folder}/model")
            # save policy weights
            stats_path = f"{iteration_folder}/stats.pth"
            # save stats for normalization
            self.original_env.save(stats_path)
            # save text file with both paths
            steepness, amplitude, frequency = self.original_env.venv.env_method("get_current_slope_params")[0]
            create_retrain_script(self.root_folder, iteration_folder, self.model_args, [steepness, amplitude, frequency])

            with open(f"{iteration_folder}/args.txt", "w") as f:
                f.write(f"python3 main.py --load --no_save --user_input  --model_path={iteration_folder} --save_video")
        if self.eval_iteration % self.model_args.eval_period == 0:
            # set slope params to max
            # self.original_env.venv.env_method("set_slope_params_for_eval")
            # Evaluate the trained agent
            # create uniform distribution from last column of self.slope_feasibility_arr
            uniform_dist = self.slope_feasibility_arr[:, -1] / np.sum(self.slope_feasibility_arr[:, -1])
            self.original_env.venv.env_method("parameterized_reset", [uniform_dist, self.slope_feasibility_arr])
            mean_reward, std_reward = evaluate_policy(self.model, self.original_env, n_eval_episodes=self.model_args.n_eval_episodes, deterministic=True)
            self.original_env.venv.env_method("parameterized_reset", [normalized_dist, self.slope_feasibility_arr])
            # self.original_env.venv.env_method("reset_after_eval")
            print(f"eval_mean_reward={mean_reward:.2f} +/- {std_reward}")
            # if mean_reward is greater than args.reward_threshold, save model and tune parameters
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
                steepness, amplitude, frequency = self.original_env.venv.env_method("get_current_slope_params")[0]
                create_retrain_script(self.root_folder, iteration_folder, self.model_args, [steepness, amplitude, frequency])


            # if mean_reward > self.model_args.reward_threshold:
            #     # in env.venv.envs
            #     # self.original_env.venv.env_method("adjust_slope_params")
            #     steepness, amplitude, frequency = self.original_env.venv.env_method("get_current_slope_params")[0]
            #     if self.model_args.use_wandb:
            #         wandb.log({"steepness": steepness, "amplitude": amplitude, "frequency": frequency, "ppo_iteration": self.iteration, "steps": self.steps})
            #     self.eval_iteration = 0
                # print("self.original_env.venv", self.original_env.envs)
                # for env in self.original_env.venv:
                #     print("ENV", env)
                #     env.adjust_slope_params()
                    
            if self.model_args.use_wandb:
                wandb.log({"eval_mean_reward": mean_reward, "std_reward": std_reward, "ppo_iteration": self.iteration, "steps": self.steps})
            # if self.model_args.save_video:
            #     self.save_video_on_training(folder=iteration_folder)
        self.iteration += 1
        self.eval_iteration += 1
        self.original_env.training = True
