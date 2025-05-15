# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import collections
import math as math_o

from deepscaler.rewards.math_reward import deepscaler_reward_fn

def _select_rm_score_fn(data_source):
    # if data_source == 'openai/gsm8k':
    #     return gsm8k.compute_score
    # elif data_source == 'lighteval/MATH':
    #     return math.compute_score
    # else:
    return deepscaler_reward_fn


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, val, alpha, beta, batch_balance_rate) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.val = val
        self.alpha = alpha
        self.beta = beta

        # ======  New parameters ======
        self.batch_balance_rate = batch_balance_rate
        self.penalty_slope = 2.0

    def _reweight_rewards(self, rewards, lengths, indices, think_flags, is_correct_flags):
        rewards = torch.tensor(rewards, dtype=torch.float32)
        lengths = torch.tensor(lengths, dtype=torch.float32)

        unique_ids = set(indices)
        for qid in unique_ids:
            group = [i for i, idx in enumerate(indices) if idx == qid]

            group_pos = [i for i in group if is_correct_flags[i]]
            group_neg = [i for i in group if not is_correct_flags[i]]

            group_think = [i for i in group if think_flags[i]]
            group_no_think = [i for i in group if not think_flags[i]]

            # Save the reward before modification
            rewards_before = rewards[group].tolist()

            # ====== 1. Reweight by length / Correct answer first.  ======
            if len(group_pos) > 1:
                group_lengths = lengths[group_pos]
                avg_len = group_lengths.mean()
                std_len = group_lengths.std(unbiased=True).clamp_min(1e-6)

                for i in group_pos:
                    z = (lengths[i] - avg_len) / std_len
                    rewards[i] = rewards[i] + ( -1 + math_o.exp(-self.alpha * z.item()) )
                # print(f"[REWEIGHT-POS-LENGTH] qid={qid}")

            if len(group_neg) > 1:
                group_lengths = lengths[group_neg]
                avg_len = group_lengths.mean()
                std_len = group_lengths.std(unbiased=True).clamp_min(1e-6)

                for i in group_neg:
                    z = (lengths[i] - avg_len) / std_len
                    rewards[i] = rewards[i] + (1 -math_o.exp(-self.beta * z.item()) )
                # print(f"[REWEIGHT-NEG-LENGTH] qid={qid}")

        # ====== 2. Dynamically adjust think/no-think rewards based on the entire batch (based on flags rather than values) ======
        rewards_before = rewards.tolist()

        num_total = len(rewards)
        num_think = sum(think_flags)
        num_no_think = num_total - num_think

        if num_total > 0:
            think_rate = num_think / num_total
            no_think_rate = num_no_think / num_total

            # -------- Process think samples (if the think ratio exceeds the threshold) --------
            if think_rate > self.batch_balance_rate:
                adjust_factor_think = (think_rate - self.batch_balance_rate) * self.penalty_slope
                adjust_factor_think = max(0.0, min(1.0, adjust_factor_think))

                for i in range(num_total):
                    if think_flags[i] and is_correct_flags[i]:  # think + correct
                        target_reward = 0
                        rewards[i] = (1 - adjust_factor_think) * rewards[i] + adjust_factor_think * target_reward
                    elif think_flags[i] and not is_correct_flags[i]:  # think + wrong
                        target_reward = -1
                        rewards[i] = (1 - adjust_factor_think) * rewards[i] + adjust_factor_think * target_reward

            # -------- Process no-think samples (if the no-think ratio exceeds the threshold) --------
            if no_think_rate > self.batch_balance_rate:
                adjust_factor_nothink = (no_think_rate - self.batch_balance_rate) * self.penalty_slope
                adjust_factor_nothink = max(0.0, min(1.0, adjust_factor_nothink))

                for i in range(num_total):
                    if not think_flags[i] and is_correct_flags[i]:  # no-think + correct
                        target_reward = 1.0
                        rewards[i] = (1 - adjust_factor_nothink) * rewards[i] + adjust_factor_nothink * target_reward
                    elif not think_flags[i] and not is_correct_flags[i]:  # no-think + wrong
                        target_reward = -2.0
                        rewards[i] = (1 - adjust_factor_nothink) * rewards[i] + adjust_factor_nothink * target_reward

        rewards_after = rewards.tolist()

        print(f"[REWEIGHT-FINAL-BATCH] think_rate={think_rate:.2f}, no_think_rate={no_think_rate:.2f}")
        print(f"  rewards(before)={rewards_before}")
        print(f"  rewards(after) ={rewards_after}")


        return rewards.tolist()


    
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch.get('data_source', "")  # 默认设为空字符串
            compute_score_fn = _select_rm_score_fn(data_source)
            reward, is_correct, think = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, val=self.val)
            
            # with print_lock:
            #     if data_source not in already_print_data_sources:
            #         already_print_data_sources[data_source] = 0

            #     if already_print_data_sources[data_source] < self.num_examine:
            #         already_print_data_sources[data_source] += 1
            #         print(sequences_str)      
            # return i, score, valid_response_length
            return i, reward, is_correct, think, valid_response_length

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=96) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        rewards, lengths, think_flags, is_correct_flags = [], [], [], []
        for i, reward, correct, think, resp_len in results:
            rewards.append(reward)
            lengths.append(resp_len.item())
            think_flags.append(think)
            is_correct_flags.append(correct)

        indices = data.non_tensor_batch.get('uid', list(range(len(data))))

        # ====== Reweight during training ======
        if not self.val:
            rewards = self._reweight_rewards(rewards, lengths, indices, think_flags, is_correct_flags)

        for i, reward in enumerate(rewards):
            reward_tensor[i, int(lengths[i]) - 1] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "think_flags": think_flags,
                    "is_correct_flags": is_correct_flags,
                }
            }
        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    print("dsr")
    print(config.get("debug_note", "config not loaded"))
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, val=False, alpha=config.data.reward_alpha, beta=config.data.reward_beta, batch_balance_rate=config.data.batch_balance_rate)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, val=True, alpha=config.data.reward_alpha, beta=config.data.reward_beta, batch_balance_rate=config.data.batch_balance_rate)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
