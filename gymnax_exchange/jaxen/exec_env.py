# from jax import config
# config.update("jax_enable_x64",True)

# ============== testing scripts ===============
from functools import partial
import jax
import jax.numpy as jnp
import gymnax
import sys
import os
sys.path.append('.')
# sys.path.append('/Users/sasrey/AlphaTrade')
# sys.path.append('/homes/80/kang/AlphaTrade')
from gymnax_exchange import utils
# from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
import chex
import timeit
import dataclasses

import faulthandler
faulthandler.enable()

print("Num Jax Devices:", jax.device_count(), "Device List:", jax.devices())

chex.assert_gpu_available(backend=None)

# #Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False)
# config.update("jax_disable_jit", True)

import random
# ============== testing scripts ===============



from ast import Dict
from contextlib import nullcontext
from email import message
from random import sample
from re import L
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, flatten_util
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
# from gymnax_exchange.test_scripts.comparison import twapV3
import time 

@struct.dataclass
class EnvState:
    ask_raw_orders: chex.Array
    bid_raw_orders: chex.Array
    trades: chex.Array
    best_asks: chex.Array
    best_bids: chex.Array
    init_time: chex.Array
    time: chex.Array
    customIDcounter: int
    window_index:int
    init_price:int
    task_to_execute:int
    quant_executed:int
    total_revenue:int
    step_counter: int
    max_steps_in_episode: int
    
    slippage_rm: int
    price_adv_rm: int
    price_drift_rm: int
    vwap_rm: int


@struct.dataclass
class EnvParams:
    is_buy_task: int
    message_data: chex.Array
    book_data: chex.Array
    stateArray_list: chex.Array
    obs_sell_list: chex.Array
    obs_buy_list: chex.Array
    episode_time: int = 60*10  # 60*30 #60seconds times 30 minutes = 1800seconds
    # max_steps_in_episode: int = 100 # TODO should be a variable, decied by the data_window
    # messages_per_step: int=1 # TODO never used, should be removed?
    time_per_step: int = 0##Going forward, assume that 0 implies not to use time step?
    time_delay_obs_act: chex.Array = jnp.array([0, 0]) #0ns time delay.
    avg_twap_list=jnp.array([312747.47,
                            312674.06,
                            313180.38,
                            312813.25,
                            312763.78,
                            313094.1,
                            313663.97,
                            313376.72,
                            313533.25,
                            313578.9,
                            314559.1,
                            315201.1,
                            315190.2])
    


class ExecutionEnv(BaseLOBEnv):
    def __init__(
            self, alphatradePath, randomize_direction, window_index, action_type,
            task_size = 500, rewardLambda=0.0, Gamma=0.00
        ):
        super().__init__(alphatradePath)
        #self.n_actions = 2 # [A, MASKED, P, MASKED] Agressive, MidPrice, Passive, Second Passive
        # self.n_actions = 2 # [MASKED, MASKED, P, PP] Agressive, MidPrice, Passive, Second Passive
        self.n_actions = 4 # [FT, M, NT, PP] Agressive, MidPrice, Passive, Second Passive
        # self.task = task
        self.randomize_direction = randomize_direction
        self.window_index = window_index
        self.action_type = action_type
        self.rewardLambda = rewardLambda
        self.Gamma = Gamma
        # self.task_size = 5000 # num to sell or buy for the task
        # self.task_size = 2000 # num to sell or buy for the task
        self.task_size = task_size # num to sell or buy for the task
        # self.task_size = 200 # num to sell or buy for the task
        self.n_fragment_max = 2
        self.n_ticks_in_book = 2 #TODO: Used to be 20, too large for stocks with dense LOBs
        # self.debug : bool = False


    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        # return EnvParams(self.messages,self.books)
        is_buy_task = 0
        return EnvParams(
            is_buy_task, self.messages, self.books, self.stateArray_list,
            self.obs_sell_list, self.obs_buy_list)
    

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, a: jax.Array, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        #Obtain the messages for the step from the message data
        # '''
        def reshape_action(action : jax.Array, state: EnvState, params : EnvParams):
            def twapV3(state, env_params):
                # ---------- ifMarketOrder ----------
                remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
                marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
                ifMarketOrder = (remainingTime <= marketOrderTime)
                # print(f"{i} remainingTime{remainingTime} marketOrderTime{marketOrderTime}")
                # ---------- ifMarketOrder ----------
                # ---------- quants ----------
                remainedQuant = state.task_to_execute - state.quant_executed
                remainedStep = state.max_steps_in_episode - state.step_counter
                stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
                limit_quants = jax.random.permutation(key, jnp.array([stepQuant-stepQuant//2,stepQuant//2]), independent=True)
                market_quants = jnp.array([stepQuant,stepQuant])
                quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
                # ---------- quants ----------
                return jnp.array(quants) 

            def truncate_action(action, remainQuant):
                action = jnp.round(action).astype(jnp.int32).clip(0, remainQuant)
                scaledAction = utils.clip_by_sum_int(action, remainQuant)
                return scaledAction

            if self.action_type == 'delta':
                action = twapV3(state, params) + action

            action = truncate_action(action, state.task_to_execute - state.quant_executed)
            # jax.debug.print("base_ {}, delta_ {}, action_ {}; action {}",base_, delta_,action_,action)
            # jax.debug.print("action {}", action)
            return action

        action = reshape_action(a, state, params)
        
        data_messages = job.get_data_messages(
            params.message_data,
            state.window_index,
            state.step_counter
        )
        #Assumes that all actions are limit orders for the moment - get all 8 fields for each action message
        
        action_msgs = self.getActionMsgs(action, state, params)
        # jax.debug.print("action_msgs {}",action_msgs)
        #Currently just naive cancellation of all agent orders in the book. #TODO avoid being sent to the back of the queue every time. 
        raw_orders = jax.lax.cond(
            params.is_buy_task,
            lambda: state.bid_raw_orders,
            lambda: state.ask_raw_orders
        )
        # jax.debug.print("raw_orders {}", raw_orders)
        cnl_msgs = job.getCancelMsgs(
            raw_orders,
            -8999,
            self.n_actions,
            params.is_buy_task * 2 - 1  # direction in {-1, 1}
        )
        # net actions and cancellations at same price if new action is not bigger than cancellation
        action_msgs, cnl_msgs = self.filter_messages(action_msgs, cnl_msgs)
        # prepend cancel and new action to data messages
        total_messages = jnp.concatenate(
            [cnl_msgs, action_msgs, data_messages],
            axis=0
        ) # TODO DO NOT FORGET TO ENABLE CANCEL MSG
        # Save time of final message to add to state
        # time=total_messages[-1:][0][-2:]
        time = total_messages[-1, -2:]
        # To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        trades_reinit = (jnp.ones((self.nTradesLogged,6))*-1).astype(jnp.int32)
        # Process messages of step (action+data) through the orderbook
        # jax.debug.breakpoint()
        asks, bids, trades, bestasks, bestbids = job.scan_through_entire_array_save_bidask(
            total_messages,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            self.stepLines
        )
        # DEBUGBEST
        # jax.debug.print("bestasks[-1] - bids[-1] {}", bestasks[-1, 0] - bestbids[-1, 0])
        # jax.debug.breakpoint()
        
        # ========== get reward and revenue ==========
        #Gather the 'trades' that are nonempty, make the rest 0
        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)
        #Mask to keep only the trades where the RL agent is involved, apply mask.
        mask2 = ((-9000 < executed[:, 2]) & (executed[:, 2] < 0)) | ((-9000 < executed[:, 3]) & (executed[:, 3] < 0))
        agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)
        #TODO: Is this truncation still needed given the action quantities will always < remaining exec quant?
        def truncate_agent_trades(agentTrades, remainQuant):
            quantities = agentTrades[:, 1]
            cumsum_quantities = jnp.cumsum(quantities)
            cut_idx = jnp.argmax(cumsum_quantities >= remainQuant)
            truncated_agentTrades = jnp.where(jnp.arange(len(quantities))[:, jnp.newaxis] > cut_idx, jnp.zeros_like(agentTrades[0]), agentTrades.at[:, 1].set(jnp.where(jnp.arange(len(quantities)) < cut_idx, quantities, jnp.where(jnp.arange(len(quantities)) == cut_idx, remainQuant - cumsum_quantities[cut_idx - 1], 0))))
            return jnp.where(remainQuant >= jnp.sum(quantities), agentTrades, jnp.where(remainQuant <= quantities[0], jnp.zeros_like(agentTrades).at[0, :].set(agentTrades[0]).at[0, 1].set(remainQuant), truncated_agentTrades))
        # agentTrades = truncate_agent_trades(agentTrades, state.task_to_execute-state.quant_executed)
        new_execution = agentTrades[:,1].sum()
        revenue = (agentTrades[:,0]//self.tick_size * agentTrades[:,1]).sum()
        agentQuant = agentTrades[:,1].sum()
        vwapFunc = lambda executed: (executed[:,0]//self.tick_size* executed[:,1]).sum()//(executed[:,1]).sum()
        vwap = vwapFunc(executed) # average_price of all the tradings, from the varaible executed
        rollingMeanValueFunc_FLOAT = lambda average_price,new_price:(average_price*state.step_counter+new_price)/(state.step_counter+1)
        rollingMeanValueFunc_INT = lambda average_price,new_price:((average_price*state.step_counter+new_price)/(state.step_counter+1)).astype(jnp.int32)
        vwap_rm = rollingMeanValueFunc_INT(state.vwap_rm,vwap) # (state.market_rap*state.step_counter+executedAveragePrice)/(state.step_counter+1)

        #TODO VWAP price (vwap) is only over all trades in between steps. 
        advantage = revenue - vwap_rm * agentQuant ### (weightedavgtradeprice-vwap)*agentQuant ### revenue = weightedavgtradeprice*agentQuant
        rewardLambda = self.rewardLambda
        drift = agentQuant * (vwap_rm - state.init_price//self.tick_size)
        # ---------- used for slippage, price_drift, and  RM(rolling mean) ----------
        price_adv_rm = rollingMeanValueFunc_INT(state.price_adv_rm,revenue//agentQuant - vwap) # slippage=revenue/agentQuant-vwap, where revenue/agentQuant means agentPrice 
        slippage_rm = rollingMeanValueFunc_INT(state.slippage_rm,revenue - state.init_price//self.tick_size*agentQuant)
        price_drift_rm = rollingMeanValueFunc_INT(state.price_drift_rm,(vwap - state.init_price//self.tick_size)) #price_drift = (vwap - state.init_price//self.tick_size)
        # ---------- compute the final reward ----------
        # rewardValue = advantage + rewardLambda * drift
        rewardValue = revenue - (state.init_price // self.tick_size) * agentQuant

        reward = jnp.sign(agentQuant) * rewardValue # if no value agentTrades then the reward is set to be zero
        # ---------- normalize the reward ----------
        reward /= 10000
        # reward /= params.avg_twap_list[state.window_index]
        # ========== get reward and revenue END ==========
        
        #Update state (ask,bid,trades,init_time,current_time,OrderID counter,window index for ep, step counter,init_price,trades to exec, trades executed)
        def bestPricesImpute(bestprices, lastBestPrice):
            def replace_values(prev, curr):
                last_non_999999999_values = jnp.where(curr != 999999999, curr, prev) #non_999999999_mask
                replaced_curr = jnp.where(curr == 999999999, last_non_999999999_values, curr)
                return last_non_999999999_values, replaced_curr
            def forward_fill_999999999_int(arr):
                last_non_999999999_values, replaced = jax.lax.scan(replace_values, arr[0], arr[1:])
                return jnp.concatenate([arr[:1], replaced])
            def forward_fill(arr):
                index = jnp.argmax(arr[:, 0] != 999999999)
                return forward_fill_999999999_int(arr.at[0, 0].set(jnp.where(index == 0, arr[0, 0], arr[index][0])))
            back_fill = lambda arr: jnp.flip(forward_fill(jnp.flip(arr, axis=0)), axis=0)
            mean_forward_back_fill = lambda arr: (forward_fill(arr)+back_fill(arr))//2
            return jnp.where(
                (bestprices[:,0] == 999999999).all(),
                jnp.tile(
                    jnp.array([lastBestPrice, 0]),
                    (bestprices.shape[0], 1)
                ),
                mean_forward_back_fill(bestprices)
            )

        bestasks = bestPricesImpute(
            bestasks[-self.stepLines:], state.best_asks[-1,0])
        bestbids = bestPricesImpute(
            bestbids[-self.stepLines:], state.best_bids[-1,0])
        state = EnvState(
            asks, bids, trades, bestasks, bestbids,
            state.init_time, time, state.customIDcounter + self.n_actions, state.window_index,
            state.init_price, state.task_to_execute, state.quant_executed + new_execution,
            state.total_revenue + revenue, state.step_counter + 1,
            state.max_steps_in_episode,
            slippage_rm, price_adv_rm, price_drift_rm, vwap_rm)
            # state.max_steps_in_episode,state.twap_total_revenue+twapRevenue,state.twap_quant_arr)
        # jax.debug.breakpoint()
        done = self.is_terminal(state, params)
        return self.get_obs(state, params), state, reward, done, {
            "window_index": state.window_index,
            "total_revenue": state.total_revenue,
            "quant_executed": state.quant_executed,
            "task_to_execute": state.task_to_execute,
            "average_price": state.total_revenue / state.quant_executed,
            "current_step": state.step_counter,
            'done': done,
            'slippage_rm': state.slippage_rm,
            "price_adv_rm": state.price_adv_rm,
            "price_drift_rm": state.price_drift_rm,
            "vwap_rm": state.vwap_rm,
            "advantage_reward": advantage,
        }
    
    def filter_messages(
            self, 
            action_msgs: jax.Array,
            cnl_msgs: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
        """ Filter out cancelation messages, when same actions should be placed again.
            NOTE: only simplifies cancellations if new action size <= old action size.
                  To prevent multiple split orders, new larger orders still cancel the entire old order.
            TODO: consider allowing multiple split orders
            ex: at one level, 3 cancel & 1 action --> 2 cancel, 0 action
        """
        @partial(jax.vmap, in_axes=(0, None))
        def p_in_cnl(p, prices_cnl):
            return jnp.where((prices_cnl == p) & (p != 0), True, False)
        def matching_masks(prices_a, prices_cnl):
            res = p_in_cnl(prices_a, prices_cnl)
            return jnp.any(res, axis=1), jnp.any(res, axis=0)
        
        # jax.debug.print("action_msgs\n {}", action_msgs)
        # jax.debug.print("cnl_msgs\n {}", cnl_msgs)

        a_mask, c_mask = matching_masks(action_msgs[:, 3], cnl_msgs[:, 3])
        # jax.debug.print("a_mask \n{}", a_mask)
        # jax.debug.print("c_mask \n{}", c_mask)
        # jax.debug.print("MASK DIFF: {}", a_mask.sum() - c_mask.sum())
        
        a_i = jnp.where(a_mask, size=a_mask.shape[0], fill_value=-1)[0]
        a = jnp.where(a_i == -1, 0, action_msgs[a_i][:, 2])
        c_i = jnp.where(c_mask, size=c_mask.shape[0], fill_value=-1)[0]
        c = jnp.where(c_i == -1, 0, cnl_msgs[c_i][:, 2])
        
        # jax.debug.print("a_i \n{}", a_i)
        # jax.debug.print("a \n{}", a)
        # jax.debug.print("c_i \n{}", c_i)
        # jax.debug.print("c \n{}", c)

        rel_cnl_quants = (c >= a) * a
        # rel_cnl_quants = jnp.maximum(0, c - a)
        # jax.debug.print("rel_cnl_quants {}", rel_cnl_quants)
        # reduce both cancel and action message quantities to simplify
        action_msgs = action_msgs.at[:, 2].set(
            action_msgs[:, 2] - rel_cnl_quants[utils.rank_rev(a_mask)])
        # set actions with 0 quant to dummy messages
        action_msgs = jnp.where(
            (action_msgs[:, 2] == 0).T,
            0,
            action_msgs.T,
        ).T
        cnl_msgs = cnl_msgs.at[:, 2].set(
            cnl_msgs[:, 2] - rel_cnl_quants[utils.rank_rev(c_mask)])
        # jax.debug.print("action_msgs NEW \n{}", action_msgs)
        # jax.debug.print("cnl_msgs NEW \n{}", cnl_msgs)

        return action_msgs, cnl_msgs

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position in OB."""
        # all windows can be reached
        
        idx_data_window = jax.random.randint(key, minval=0, maxval=self.n_windows, shape=()) if self.window_index == -1 else jnp.array(self.window_index,dtype=jnp.int32)
        # idx_data_window = jnp.array(self.window_index,dtype=jnp.int32)
        # one window can be reached
        
        # jax.debug.print("window_size {}",self.max_steps_in_episode_arr[0])
        
        # task_size,content_size,array_size = self.task_size,self.max_steps_in_episode_arr[idx_data_window],self.max_steps_in_episode_arr.max().astype(jnp.int32) 
        # task_size,content_size,array_size = self.task_size,self.max_steps_in_episode_arr[idx_data_window],1000
        # base_allocation = task_size // content_size
        # remaining_tasks = task_size % content_size
        # array = jnp.full(array_size, 0, dtype=jnp.int32)
        # array = array.at[:remaining_tasks].set(base_allocation+1)
        # twap_quant_arr = array.at[remaining_tasks:content_size].set(base_allocation)
        
        def stateArray2state(stateArray):
            state0 = stateArray[:,0:6];state1 = stateArray[:,6:12];state2 = stateArray[:,12:18];state3 = stateArray[:,18:20];state4 = stateArray[:,20:22]
            state5 = stateArray[0:2,22:23].squeeze(axis=-1);state6 = stateArray[2:4,22:23].squeeze(axis=-1);state9= stateArray[4:5,22:23][0].squeeze(axis=-1)
            return (state0,state1,state2,state3,state4,state5,state6,0,idx_data_window,state9,self.task_size,0,0,0,self.max_steps_in_episode_arr[idx_data_window],0,0,0,0)
            # return (state0,state1,state2,state3,state4,state5,state6,0,idx_data_window,state9,self.task_size,0,0,0,self.max_steps_in_episode_arr[idx_data_window])
            # return (state0,state1,state2,state3,state4,state5,state6,0,idx_data_window,state9,self.task_size,0,0,0,self.max_steps_in_episode_arr[idx_data_window],0,twap_quant_arr)
        key_, key = jax.random.split(key)
        if self.randomize_direction:
            direction = jax.random.randint(key_, minval=0, maxval=2, shape=())
            params = dataclasses.replace(params, is_buy_task=direction)
        stateArray = params.stateArray_list[idx_data_window]
        state_ = stateArray2state(stateArray)
        # print(self.max_steps_in_episode_arr[idx_data_window])
        obs_sell = params.obs_sell_list[idx_data_window]
        obs_buy = params.obs_buy_list[idx_data_window]
        state = EnvState(*state_)
        # jax.debug.print('in reset_env')
        # jax.debug.print('{}', job.get_L2_state(state.ask_raw_orders, state.bid_raw_orders, 3))
        # jax.debug.breakpoint()
        # jax.debug.print("state after reset {}", state)
        # why would we use the observation from the parent env here?
        # obs = obs_sell if self.task == "sell" else obs_buy
        obs = self.get_obs(state, params)

        return obs, state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return (
            ((state.time - state.init_time)[0] > params.episode_time) |
            (state.task_to_execute - state.quant_executed <= 0)
        )
    
    def getActionMsgs(self, action: jax.Array, state: EnvState, params: EnvParams):
        # def normal_order_logic(action: jnp.ndarray):
        #     quants = action.astype(jnp.int32) # from action space
        #     return quants

        # def market_order_logic(state: EnvState):
        #     quant = state.task_to_execute - state.quant_executed
        #     quants = jnp.asarray((quant, 0, 0, 0), jnp.int32) 
        #     return quants
        
        def normal_qp(price_levels: jax.Array, state: EnvState, action: jax.Array):
            def combine_mid_nt(quants, prices):
                quants = quants \
                    .at[2].set(quants[2] + quants[1]) \
                    .at[1].set(0)
                prices = prices.at[1].set(-1)
                return quants, prices

            quants = action.astype(jnp.int32)
            prices = jnp.array(price_levels[:-1])
            # if mid_price == near_touch_price: combine orders into one
            return jax.lax.cond(
                price_levels[1] != price_levels[2],
                lambda q, p: (q, p),
                combine_mid_nt,
                quants, prices
            )
        
        def market_qp(price_levels: jax.Array, state: EnvState, action: jax.Array):
            mkt_quant = state.task_to_execute - state.quant_executed
            quants = jnp.asarray((mkt_quant, 0, 0, 0), jnp.int32) 
            return quants, jnp.asarray((price_levels[-1], -1, -1, -1), jnp.int32)
        
        def buy_task_prices(best_ask, best_bid):
            NT = best_bid
            # mid defaults to one tick more passive if between ticks
            M = ((best_bid + best_ask) // 2 // self.tick_size) * self.tick_size
            FT = best_ask
            PP = best_bid - self.tick_size*self.n_ticks_in_book
            MKT = job.MAX_INT
            return NT, M, FT, PP, MKT

        def sell_task_prices(best_ask, best_bid):
            NT = best_ask
            # mid defaults to one tick more passive if between ticks
            M = (jnp.ceil((best_bid + best_ask) / 2 / self.tick_size)
                 * self.tick_size).astype(jnp.int32)
            FT = best_bid
            PP = best_ask + self.tick_size*self.n_ticks_in_book
            MKT = 0
            return NT, M, FT, PP, MKT

        # ============================== Get Action_msgs ==============================
        # --------------- 01 rest info for deciding action_msgs ---------------
        types = jnp.ones((self.n_actions,), jnp.int32)
        # sides=-1*jnp.ones((self.n_actions,),jnp.int32) if self.task=='sell' else jnp.ones((self.n_actions),jnp.int32) #if self.task=='buy'
        sides = (params.is_buy_task*2 - 1) * jnp.ones((self.n_actions,), jnp.int32)
        trader_ids = jnp.ones((self.n_actions,), jnp.int32) * self.trader_unique_id #This agent will always have the same (unique) trader ID
        order_ids = (jnp.ones((self.n_actions,), jnp.int32) *
                    (self.trader_unique_id + state.customIDcounter)) \
                    + jnp.arange(0, self.n_actions) #Each message has a unique ID
        times = jnp.resize(
            state.time + params.time_delay_obs_act,
            (self.n_actions, 2)
        ) #time from last (data) message of prev. step + some delay
        #Stack (Concatenate) the info into an array 
        # --------------- 01 rest info for deciding action_msgs ---------------
        
        # --------------- 02 info for deciding prices ---------------
        # Can only use these if statements because self is a static arg.
        # Done: We said we would do ticks, not levels, so really only the best bid/ask is required -- Write a function to only get those rather than sort the whole array (get_L2) 
        best_ask, best_bid = state.best_asks[-1, 0], state.best_bids[-1, 0]
        # jax.debug.print("ask - bid {}", best_ask - best_bid)

        NT, M, FT, PP, MKT = jax.lax.cond(
            params.is_buy_task,
            buy_task_prices,
            sell_task_prices,
            best_ask, best_bid
        )
        # --------------- 02 info for deciding prices ---------------

        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        remainingTime = params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
        marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order

        price_levels = (FT, M, NT, PP, MKT)
        quants, prices = jax.lax.cond(
            (remainingTime <= marketOrderTime),
            market_qp,
            normal_qp,
            price_levels, state, action
        )
        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        action_msgs = jnp.stack([types, sides, quants, prices, trader_ids, order_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs,times],axis=1)
        return action_msgs
        # ============================== Get Action_msgs ==============================

    def get_obs(self, state: EnvState, params:EnvParams) -> chex.Array:
        """Return observation from raw state trafo."""
        # NOTE: currently only uses most recent observation from state
        # quote_aggr = state.best_bids[-1] if self.task=='sell' else state.best_asks[-1]
        # quote_pass = state.best_asks[-1] if self.task=='sell' else state.best_bids[-1]
        quote_aggr, quote_pass = jax.lax.cond(
            params.is_buy_task,
            lambda: (state.best_asks[-1], state.best_bids[-1]),
            lambda: (state.best_bids[-1], state.best_asks[-1])
        )
        obs = {
            # "is_buy_task": 0. if self.task=='sell' else 1.,
            "is_buy_task": params.is_buy_task,
            "p_aggr": quote_aggr[0],
            "p_pass": quote_pass[0],
            "spread": jnp.abs(quote_aggr[0] - quote_pass[0]),
            "q_aggr": quote_aggr[1],
            "q_pass": quote_pass[1],
            # TODO: add "q_pass2" as passive quantity to state in step_env and here
            "time": state.time,
            "episode_time": state.time - state.init_time,
            "init_price": state.init_price,
            "task_size": state.task_to_execute,
            "executed_quant": state.quant_executed,
            "step_counter": state.step_counter,
            "max_steps": state.max_steps_in_episode,
        }

        def normalize_obs(obs: dict[str, jax.Array]):
            """ normalized observation by substracting 'mean' and dividing by 'std'
                (config values don't need to be actual mean and std)
            """
            # TODO: put this into config somewhere?
            #       also check if we can get rid of manual normalization
            #       by e.g. functional transformations or maybe gymnax obs norm wrapper suffices?
            p_mean = 3.5e7
            p_std = 1e6
            means = {
                "is_buy_task": 0,
                "p_aggr": p_mean,
                "p_pass": p_mean,
                "spread": 0,
                "q_aggr": 0,
                "q_pass": 0,
                "time": jnp.array([0, 0]),
                "episode_time": jnp.array([0, 0]),
                "init_price": p_mean,
                "task_size": 0,
                "executed_quant": 0,
                "step_counter": 0,
                "max_steps": 0,
            }
            stds = {
                "is_buy_task": 1,
                "p_aggr": p_std,
                "p_pass": p_std,
                "spread": 1e4,
                "q_aggr": 100,
                "q_pass": 100,
                "time": jnp.array([1e5, 1e9]),
                "episode_time": jnp.array([1e3, 1e9]),
                "init_price": p_std,
                "task_size": 500,
                "executed_quant": 500,
                "step_counter": 300,
                "max_steps": 300,
            }
            obs = jax.tree_map(lambda x, m, s: (x - m) / s, obs, means, stds)
            return obs

        obs = normalize_obs(obs)
        # jax.debug.print("obs {}", obs)
        obs, _ = jax.flatten_util.ravel_pytree(obs)
        
        return obs


    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """ Action space of the environment. """
        # return spaces.Box(-100,100,(self.n_actions,),dtype=jnp.int32) if self.action_type=='delta' else spaces.Box(0,500,(self.n_actions,),dtype=jnp.int32)
        if self.action_type == 'delta':
            return spaces.Box(-100, 100, (self.n_actions,), dtype=jnp.int32)
        else:
            return spaces.Box(0, self.task_size, (self.n_actions,), dtype=jnp.int32)

    #FIXME: Obsevation space is a single array with hard-coded shape (based on get_obs function): make this better.
    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        space = spaces.Box(-10,10,(15,),dtype=jnp.float32) 
        return space

    #FIXME:Currently this will sample absolute gibberish. Might need to subdivide the 6 (resp 5) 
    #           fields in the bid/ask arrays to return something of value. Not sure if actually needed.   
    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "bids": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "asks": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "trades": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nTradesLogged),dtype=jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "alphatradeExec-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions

# ============================================================================= #
# ============================================================================= #
# ================================== MAIN ===================================== #
# ============================================================================= #
# ============================================================================= #

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "sell",
        "TASK_SIZE": 500,
        "WINDOW_INDEX": 2,
        "ACTION_TYPE": "pure",
        "REWARD_LAMBDA": 1.0,
    }
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # env=ExecutionEnv(ATFolder,"sell",1)
    env= ExecutionEnv(
        config["ATFOLDER"],
        config["TASKSIDE"],
        config["WINDOW_INDEX"],
        config["ACTION_TYPE"],
        config["TASK_SIZE"],
        config["REWARD_LAMBDA"]
    )
    env_params=env.default_params
    # print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("Time for reset: \n",time.time()-start)
    # print("State after reset: \n",state)
   

    # print(env_params.message_data.shape, env_params.book_data.shape)
    for i in range(1,100000):
        # ==================== ACTION ====================
        # ---------- acion from random sampling ----------
        # print("-"*20)
        key_policy, _ =  jax.random.split(key_policy, 2)
        # test_action=env.action_space().sample(key_policy)
        # test_action=env.action_space().sample(key_policy) * random.randint(1, 10) # CAUTION not real action
        test_action = env.action_space().sample(key_policy) // 10
        print(f"Sampled {i}th actions are: ", test_action)
        start=time.time()
        obs, state, reward, done, info = env.step(key_step, state,test_action, env_params)
        # for key, value in info.items():
        #     print(key, value)
        # print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
        # print(f"Time for {i} step: \n",time.time()-start)
        if done:
            print("==="*20)
        # ---------- acion from random sampling ----------
        # ==================== ACTION ====================




    # # ####### Testing the vmap abilities ########
    
    enable_vmap=False
    if enable_vmap:
        # with jax.profiler.trace("/homes/80/kang/AlphaTrade/wandb/jax-trace"):
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))

        num_envs = 10
        vmap_keys = jax.random.split(rng, num_envs)

        test_actions=vmap_act_sample(vmap_keys)
        print(test_actions)

        start=time.time()
        obs, state = vmap_reset(vmap_keys, env_params)
        print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)

        start=time.time()
        n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)
        print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)
