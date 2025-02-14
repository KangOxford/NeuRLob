import os, sys, time, dataclasses
from typing import Tuple, Optional, Dict

import jax
import jax.numpy as jnp
import chex
from flax import struct
import jax.tree_util as jtu

from mm_env import MarketMakingEnv, EnvState as MMAgentState, EnvParams as MMParams
from exec_env import ExecutionEnv, EnvState as EXEAgentState, EnvParams as EXEParams
from gymnax_exchange.jaxen.base_env import BaseLOBEnv, EnvState as BaseState, EnvParams as BaseParams

@struct.dataclass
class MultiAgentState(BaseState):
    shared: BaseState
    mm_agent_states: any
    exe_agent_states: any

@struct.dataclass
class MultiAgentParams(BaseParams):
    mm_params: any
    exe_params: any

class MARLEnv(BaseLOBEnv):
    def __init__(self,
                 alphatradePath: str,
                 window_index: int,
                 episode_time: int,
                 ep_type: str = "fixed_time",
                 mm_trader_id: int = 1111,
                 exe_trader_id: int = 2222,
                 mm_reward_lambda: float = 0.0001,
                 exe_reward_lambda: float = 1.0,
                 exe_task_size: int = 100,
                 mm_action_type: str = "pure",
                 mm_n_ticks_in_book: int = 2,
                 mm_max_task_size: int = 500,
                 num_mm_agents: int = 1,
                 num_exe_agents: int = 1):
        super().__init__(alphatradePath, window_index, episode_time, ep_type=ep_type)
        self.mm_env = MarketMakingEnv(
            alphatradePath,
            window_index=window_index,
            action_type=mm_action_type,
            episode_time=episode_time,
            max_task_size=mm_max_task_size,
            rewardLambda=mm_reward_lambda,
            ep_type=ep_type
        )
        self.exe_env = ExecutionEnv(
            alphatradePath,
            task="buy",
            window_index=window_index,
            action_type="pure",
            episode_time=episode_time,
            max_task_size=exe_task_size,
            rewardLambda=exe_reward_lambda,
            ep_type=ep_type
        )
        self.mm_trader_id = mm_trader_id
        self.exe_trader_id = exe_trader_id
        self.num_mm_agents = num_mm_agents
        self.num_exe_agents = num_exe_agents
        self.mm_agent_names = [f"mm_{i}" for i in range(self.num_mm_agents)]
        self.exe_agent_names = [f"exe_{i}" for i in range(self.num_exe_agents)]
        self.agents = self.mm_agent_names + self.exe_agent_names

    def default_params(self) -> MultiAgentParams:
        base_params = super().default_params()
        mm_single = self.mm_env.default_params()
        exe_single = self.exe_env.default_params()
        mm_batched = jax.tree_map(lambda x: jnp.repeat(x[None, ...], self.num_mm_agents, axis=0),
                                  dataclasses.asdict(mm_single))
        exe_batched = jax.tree_map(lambda x: jnp.repeat(x[None, ...], self.num_exe_agents, axis=0),
                                   dataclasses.asdict(exe_single))
        return MultiAgentParams(
            **dataclasses.asdict(base_params),
            mm_params=mm_batched,
            exe_params=exe_batched
        )

    @jax.jit
    def reset_env(self, key: chex.PRNGKey, params: MultiAgentParams) -> Tuple[Dict[str, jnp.ndarray], MultiAgentState]:
        base_obs, shared_state = self.mm_env.reset_env(key, jax.tree_map(lambda x: x[0], params.mm_params))
        mm_keys = jax.random.split(key, self.num_mm_agents)
        i_arr = jnp.arange(self.num_mm_agents)
        mm_out = jax.vmap(lambda k, i: self.mm_env.reset_env(k, jax.tree_map(lambda x: x[i], params.mm_params)), in_axes=(0, 0))(mm_keys, i_arr)
        mm_obs_batch, mm_states_batch = mm_out
        exe_keys = jax.random.split(key, self.num_exe_agents)
        i_arr_exe = jnp.arange(self.num_exe_agents)
        exe_out = jax.vmap(lambda k, i: self.exe_env.reset_env(k, jax.tree_map(lambda x: x[i], params.exe_params)), in_axes=(0, 0))(exe_keys, i_arr_exe)
        exe_obs_batch, exe_states_batch = exe_out
        mm_obs_dict = {name: mm_obs_batch[i] for i, name in enumerate(self.mm_agent_names)}
        exe_obs_dict = {name: exe_obs_batch[i] for i, name in enumerate(self.exe_agent_names)}
        obs = {**mm_obs_dict, **exe_obs_dict}
        multi_state = MultiAgentState(
            shared=shared_state,
            mm_agent_states=mm_states_batch,
            exe_agent_states=exe_states_batch
        )
        return obs, multi_state

    @jax.jit
    def step_env(self,
                 key: chex.PRNGKey,
                 state: MultiAgentState,
                 actions: Dict[str, jnp.ndarray],
                 params: MultiAgentParams
                 ) -> Tuple[Dict[str, jnp.ndarray], MultiAgentState, Dict[str, float], bool, Dict[str, Dict]]:
        mm_actions = jnp.stack([actions[name] for name in self.mm_agent_names], axis=0)
        exe_actions = jnp.stack([actions[name] for name in self.exe_agent_names], axis=0)
        key_mm, key_exe, key_shared = jax.random.split(key, 3)
        i_arr = jnp.arange(self.num_mm_agents)
        mm_out = jax.vmap(lambda k, s, a, i: self.mm_env.step_env(k, s, a, jax.tree_map(lambda x: x[i], params.mm_params)), in_axes=(0, 0, 0, 0))(jax.random.split(key_mm, self.num_mm_agents), state.mm_agent_states, mm_actions, i_arr)
        mm_obs_batch, new_mm_states, mm_rewards, mm_done, mm_info = mm_out
        i_arr_exe = jnp.arange(self.num_exe_agents)
        exe_out = jax.vmap(lambda k, s, a, i: self.exe_env.step_env(k, s, a, jax.tree_map(lambda x: x[i], params.exe_params)), in_axes=(0, 0, 0, 0))(jax.random.split(key_exe, self.num_exe_agents), state.exe_agent_states, exe_actions, i_arr_exe)
        exe_obs_batch, new_exe_states, exe_rewards, exe_done, exe_info = exe_out
        new_shared_state = self._update_shared_state(state.shared, key_shared, params)
        mm_obs_dict = {name: mm_obs_batch[i] for i, name in enumerate(self.mm_agent_names)}
        exe_obs_dict = {name: exe_obs_batch[i] for i, name in enumerate(self.exe_agent_names)}
        obs = {**mm_obs_dict, **exe_obs_dict}
        mm_rewards_dict = {name: float(mm_rewards[i]) for i, name in enumerate(self.mm_agent_names)}
        exe_rewards_dict = {name: float(exe_rewards[i]) for i, name in enumerate(self.exe_agent_names)}
        rewards = {**mm_rewards_dict, **exe_rewards_dict}
        done = jnp.all(jnp.concatenate([mm_done, exe_done]))
        info = {"market_maker": mm_info, "execution": exe_info}
        new_state = MultiAgentState(
            shared=new_shared_state,
            mm_agent_states=new_mm_states,
            exe_agent_states=new_exe_states
        )
        return obs, new_state, rewards, done, info

    def _update_shared_state(self, shared: BaseState, key: chex.PRNGKey, params: MultiAgentParams) -> BaseState:
        return shared

    def action_space(self, params: Optional[MultiAgentParams] = None):
        mm_space = self.mm_env.action_space(params.mm_params[0] if params is not None else None)
        exe_space = self.exe_env.action_space(params.exe_params[0] if params is not None else None)
        return {"market_maker": mm_space, "execution": exe_space}

    def observation_space(self, params: Optional[MultiAgentParams] = None):
        mm_space = self.mm_env.observation_space(params.mm_params[0] if params is not None else None)
        exe_space = self.exe_env.observation_space(params.exe_params[0] if params is not None else None)
        return {"market_maker": mm_space, "execution": exe_space}

if __name__ == "__main__":
    import sys, time, dataclasses
    try:
        ATFolder = sys.argv[1]
    except Exception:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"
    config = {
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 300,
        "WINDOW_INDEX": 1,
        "MM_TRADER_ID": 1111,
        "MM_REWARD_LAMBDA": 0.0001,
        "MM_ACTION_TYPE": "pure",
        "MM_MAX_TASK_SIZE": 500,
        "EXE_TRADER_ID": 2222,
        "EXE_REWARD_LAMBDA": 1.0,
        "EXE_TASK_SIZE": 100,
        "NUM_MM_AGENTS": 2,
        "NUM_EXE_AGENTS": 3,
    }
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    env = MARLEnv(
        alphatradePath=ATFolder,
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        ep_type=config["EP_TYPE"],
        mm_trader_id=config["MM_TRADER_ID"],
        exe_trader_id=config["EXE_TRADER_ID"],
        mm_reward_lambda=config["MM_REWARD_LAMBDA"],
        exe_reward_lambda=config["EXE_REWARD_LAMBDA"],
        exe_task_size=config["EXE_TASK_SIZE"],
        mm_action_type=config["MM_ACTION_TYPE"],
        mm_max_task_size=config["MM_MAX_TASK_SIZE"],
        num_mm_agents=config["NUM_MM_AGENTS"],
        num_exe_agents=config["NUM_EXE_AGENTS"]
    )
    env_params = env.default_params()
    env_params = dataclasses.replace(env_params, episode_time=config["EPISODE_TIME"])
    obs, state = env.reset_env(key_reset, env_params)
    for agent, o in obs.items():
        print(f"{agent}: {o}")
    for i in range(1, 20):
        print("=" * 40)
        action_mm = env.mm_env.action_space().sample(key_policy)
        action_exe = env.exe_env.action_space().sample(key_policy)
        actions = {}
        for agent in env.mm_agent_names:
            actions[agent] = action_mm
        for agent in env.exe_agent_names:
            actions[agent] = action_exe
        obs, state, rewards, done, info = env.step_env(key_step, state, actions, env_params)
        print("Step rewards:", rewards)
        print("Step info:", info)
        if done:
            print("Episode finished!")
            break
