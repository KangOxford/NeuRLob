# %% ==========================================================================
# clear warnings
import warnings
import gym
from sb3_contrib import RecurrentPPO
# from stable_baselines3 import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_trading.envs.base_environment import BaseEnv
from gym_trading.data.data_pipeline import ExternalData

warnings.filterwarnings("ignore")
Flow = ExternalData.get_sample_order_book_data()

env = gym.make("GymTrading-v1",Flow = Flow) 
check_env(env)

num_cpu = 10 
venv = DummyVecEnv([lambda: Monitor(gym.make("GymTrading-v1",Flow = Flow))] * num_cpu)
# %%
def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func

model = RecurrentPPO(
    "MlpLstmPolicy", 
    venv, 
    verbose=1,
    learning_rate = linear_schedule(1e-4),
    tensorboard_log="/Users/kang/GitHub/NeuralLOB/venv_rnn/")

model.learn(total_timesteps=int(1e5), tb_log_name="RNN_PPO_init")
model.learn(total_timesteps=int(1e12), tb_log_name="RNN_PPO_stable", reset_num_timesteps=None)
model.save("/Users/kang/GitHub/NeuralLOB/tensorboard_rnn/rnn_ppo_gym_trading-v1")

# tensorboard --logdir /Users/kang/GitHub/NeuralLOB/venv_rnn/

# %% test the train result
import time
start = time.time()
env = gym.make("GymTrading-v1",Flow = Flow) ## TODO

for i in range(int(1e3)):
    obs = env.reset()
    # ;print(obs)
    done = False
    running_reward = 0
    for i in range(int(1e8)):
        if i//int(1e5) == i/int(1e5):
            print("Epoch {}, training time {}".format(i,time.time()-start))
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        running_reward += reward 
        if done:
            running_reward += reward
            obs = env.reset()
            break 


# %% get the result
def get_result(name):
    import re  
    fp = open("/Users/kang/GitHub/NeuralLOB/FINALREMAINING(RL)")   
    # fp = open("/Users/kang/Desktop/FINALREMAINING(RL)")   
    lst = []
    for line in fp.readlines():
        try:
            m = re.search(name, line)
            result = line[m.end()+3:-1]
            lst.append(float(result))
        except:
            pass
    fp.close()
    return lst

Init = get_result('Init')
Diff = get_result('Diff')
RL = [x+y for x,y in zip(Init,Diff)]
import numpy as np
print("Diff, ",np.mean(Diff))
print("Init, ",np.mean(Init))
print("RL, ",np.mean(RL))


# %%


    