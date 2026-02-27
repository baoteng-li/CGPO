import numpy as np
import torch
from collections import deque


class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()
        self.threshold = 0.5

    # exp reward is for rwr
    def update(self, prompts, rewards, exp=False):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)*0.0
        n_rows = rewards.shape[0]
        selection_marker = []
        prompt_rewards_list = []
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # Fix: Recalculate prompt_rewards for each prompt
            # print(prompt_rewards.shape)
            # print(prompt_rewards[:2])
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
            # print(prompt_rewards.shape)
            # print(advantages.shape)
            selection_rewards_variance = self.curriculum_filter(prompt_rewards)
            selection_marker.append((prompt, selection_rewards_variance))
            prompt_rewards_list.append((prompt, prompt_rewards))
        return advantages, selection_marker, prompt_rewards_list
    
    def curriculum_filter(self, prompt_rewards, temperature=1.0):
        # 直接采样
        # rewards_means = prompt_rewards.mean(axis=1)
        # rewards_variance = np.var(rewards_means, ddof=1)
        # # print(rewards_variance)
        # selection_marker_prompt = rewards_variance > self.threshold
        # selection_marker_prompt = selection_marker_prompt.astype(int)
        # # print(selection_marker_prompt)

        # 概率采样
        rewards_means = prompt_rewards.mean(axis=1)
        rewards_variance = np.var(rewards_means, ddof=1)
        # print('rewards_variance', rewards_variance)
        selection_marker_prompt = rewards_variance / temperature
        # print('selection_marker_prompt', selection_marker_prompt)

        return selection_marker_prompt

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts
    
    def clear(self):
        self.stats = {}

def main():
    tracker = PerPromptStatTracker()
    prompts = ['a', 'b', 'a', 'c', 'b', 'a']
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)

if __name__ == "__main__":
    main()
