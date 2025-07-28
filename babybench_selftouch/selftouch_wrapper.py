# babybench_selftouch/selftouch_wrapper.py

import gymnasium as gym
import numpy as np
import torch

class TouchRewardWrapper(gym.Wrapper):
    """
    A wrapper that calculates touch-based rewards with sophisticated, part-specific timing.
    - General body touches use a diversity-driven reward (softmax+decay) gated by a standard "Reward Window" and "Cooldown".
    - Hand touches use a fixed bonus reward, gated by its own separate window and cooldown parameters.
    """
    def __init__(self, env, reward_module, 
                 # General touch parameters
                 general_reward_window=80, general_cooldown_period=100, 
                 # Hand-specific touch parameters
                 hand_reward_value=5,
                 hand_reward_window=200, hand_cooldown_period=40,
                 hand_overhold_threshold=25, 
                 hand_overhold_penalty=1.0,                
                 touch_threshold=1e-6,
                 body_idx_map=None,
                 hand_body_ids=None,
                 num_parts=22):
        super().__init__(env)
        self.reward_mod = reward_module # This module now only manages non-hand parts
        self.body_idx_map = body_idx_map
        self.touch_threshold = touch_threshold
        
        # Store separate timing parameters for hands vs. other body parts
        self.general_reward_window = general_reward_window
        self.general_cooldown_period = general_cooldown_period

        self.hand_reward_value = hand_reward_value
        self.hand_reward_window = hand_reward_window
        self.hand_cooldown_period = hand_cooldown_period
        self.hand_overhold_threshold = hand_overhold_threshold
        self.hand_overhold_penalty = hand_overhold_penalty
        
        # --- 【最终的、正确的映射逻辑】 ---
        print("--- TouchRewardWrapper: Initializing sensor-to-body mapping (Corrected Version) ---")
        
        # 1. 确定带传感器的身体部位的顺序和总数
        self.sensor_body_ids = list(env.touch.sensor_positions.keys())
        self.num_parts = len(self.sensor_body_ids)
        self.body_id_to_idx = {body_id: i for i, body_id in enumerate(self.sensor_body_ids)}

        # 2. 【关键修正】: 遍历有序列表，精确计算每个身体部位在 obs['touch'] 中的数据切片
        self.sensor_slices = {}
        current_idx_offset = 0
        
        for body_id in self.sensor_body_ids:
            # 获取该部位上的传感器数量，这直接对应于其在obs['touch']中的数据长度
            num_values_for_body = len(env.touch.sensor_positions[body_id])
            #print(f"Body ID: {body_id}, Sensor Count: {num_values_for_body}, Slice: {current_idx_offset}:{current_idx_offset + num_values_for_body}")

            # 定义该部位的数据切片
            self.sensor_slices[body_id] = slice(current_idx_offset, current_idx_offset + num_values_for_body)
            
            # 更新下一个部位的起始下标
            current_idx_offset += num_values_for_body
        
        print(f"Mapped {self.num_parts} body parts with touch sensors.")
        
        # 3. 根据传入的 hand_body_ids 确定手部和身体的【内部索引】
        self.hand_parts_indices = {self.body_id_to_idx[bid] for bid in hand_body_ids if bid in self.body_id_to_idx}
        self.body_parts_indices = set(range(self.num_parts)) - self.hand_parts_indices
        
        # print hand_parts_indices and body_parts_indices
        #print(f"Hand parts indices: {self.hand_parts_indices}")

        # 4. 创建 body_idx_map 供 reward_mod 使用
        body_indices_list_sorted = sorted(list(self.body_parts_indices))
        self.body_idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(body_indices_list_sorted)}
        
        # 初始化内部状态数组
        self.current_touch_durations = np.zeros(self.num_parts, dtype=np.int32)
        self.touch_cooldown_timers = np.zeros(self.num_parts, dtype=np.int32)
        self.parts_in_contact_last_step = set()

        # 使用正确的身体部位数量重新配置reward_mod
        self.reward_mod.reconfigure(num_parts=len(self.body_parts_indices))
        print("----------------------------------------------------------------------")



    def reset(self, **kwargs):
        """Resets the environment and all stateful trackers."""
        self.reward_mod.reset()
        self.current_touch_durations.fill(0)
        self.touch_cooldown_timers.fill(0)
        self.parts_in_contact_last_step = set()
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Performs a step and calculates rewards using the dual-path timing logic.
        """
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        # 1. Decrement all active cooldown timers.
        self.touch_cooldown_timers = np.maximum(0, self.touch_cooldown_timers - 1)

        # 2. Identify all parts currently in physical contact.
        touched_array = obs['touch']
         # --- 【关键修正】: 初始化为空集合，并删除旧的、错误的检测逻辑 ---
        parts_in_contact_now = set()
        # ----------------------------------------------------------

        
        # --- 【全新触摸检测逻辑】 ---
        for body_id in self.sensor_body_ids:
            sensor_slice = self.sensor_slices[body_id]
            sensor_readings = touched_array[sensor_slice]
            
            # 检查这个部位的任何传感器是否有读数
            if np.any(sensor_readings > self.touch_threshold):
                part_idx = self.body_id_to_idx[body_id]
                parts_in_contact_now.add(part_idx)

        # 3. Update the continuous touch duration for each part.
        is_in_contact_mask = np.zeros(self.num_parts, dtype=bool)
        if parts_in_contact_now:
            is_in_contact_mask[list(parts_in_contact_now)] = True
        
        self.current_touch_durations[is_in_contact_mask] += 1
        self.current_touch_durations[~is_in_contact_mask] = 0

        # 4. Identify released parts and set their specific cooldowns.
        released_parts = self.parts_in_contact_last_step - parts_in_contact_now
        for part_idx in released_parts:
            if part_idx in self.hand_parts_indices:
                self.touch_cooldown_timers[part_idx] = self.hand_cooldown_period
            else:
                self.touch_cooldown_timers[part_idx] = self.general_cooldown_period

        # 5. Calculate rewards based on part type (hand vs. general).
        touch_reward = 0.0
        hand_touch_reward = 0.0
        
        potential_body_rewards = self.reward_mod.compute_rewards()
        
        eligible_body_parts_for_update = []

        for part_idx in parts_in_contact_now:
            if self.touch_cooldown_timers[part_idx] == 0:
                duration = self.current_touch_durations[part_idx]
                
                if part_idx in self.hand_parts_indices:
                    if 0 < duration <= self.hand_reward_window:
                        normalized_duration = duration / self.hand_reward_window
                        hand_touch_reward += self.hand_reward_value
                    elif duration > self.hand_overhold_threshold:
                        hand_touch_reward -= self.hand_overhold_penalty
                else: # This is a non-hand body part
                    if 0 < duration <= self.general_reward_window:
                        # Use the map to get the correct local index for the reward module
                        if part_idx in self.body_idx_map:
                            local_idx = self.body_idx_map[part_idx]
                            touch_reward += potential_body_rewards[local_idx]
                    
                    if part_idx in self.body_idx_map:
                        local_idx = self.body_idx_map[part_idx]
                        eligible_body_parts_for_update.append(local_idx)

        # 6. Update the diversity counter with ONLY non-hand touched parts.
        if eligible_body_parts_for_update:
            self.reward_mod.update(eligible_body_parts_for_update)
        
        # 7. Update the state for the next step.
        self.parts_in_contact_last_step = parts_in_contact_now
        
        # --- Final reward calculation with separate lambdas ---
        total_reward = extrinsic_reward + touch_reward + hand_touch_reward
        

        if "reward_components" not in info:
            info["reward_components"] = {}
        # 现在info中只存放未加权的原始值
        info["reward_components"].update({
            'unweighted_touch': touch_reward,
            'unweighted_hand': hand_touch_reward,
        })

        return obs, total_reward, terminated, truncated, info

