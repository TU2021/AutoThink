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
Generate responses given a dataset of prompts
"""
import csv
import ray
import numpy as np
import hydra
import os
from tabulate import tabulate
from tqdm import tqdm

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

def pad_to_world_size(data: DataProto, world_size: int) -> tuple[DataProto, int]:
    real_batch_size = data.batch['input_ids'].shape[0]

    if real_batch_size < world_size:
        dummy_data_size = world_size - real_batch_size
    else:
        remainder = real_batch_size % world_size
        dummy_data_size = world_size - remainder if remainder != 0 else 0

    if dummy_data_size > 0:
        dummy_input_ids = data.batch['input_ids'][:1].repeat(dummy_data_size, 1)
        dummy_attention_mask = data.batch['attention_mask'][:1].repeat(dummy_data_size, 1)
        dummy_position_ids = data.batch['position_ids'][:1].repeat(dummy_data_size, 1)
        dummy_data = DataProto.from_dict({
            'input_ids': dummy_input_ids,
            'attention_mask': dummy_attention_mask,
            'position_ids': dummy_position_ids
        })
        data = DataProto.concat([data, dummy_data])
        print(f'[PAD] world_size={world_size}, real_batch_size={real_batch_size}, padded={dummy_data_size}')

    return data, real_batch_size



@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # Check if output file already exists
    if os.path.exists(config.data.output_path):
        print(f"Output file {config.data.output_path} already exists. Skipping generation and proceeding to evaluation.")
        dataset = pd.read_parquet(config.data.output_path)
    else:
        local_path = copy_local_path_from_hdfs(config.model.path)
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(local_path)

        if config.rollout.temperature == 0.:
            assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

        # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
        dataset = pd.read_parquet(config.data.path)
        chat_lst = dataset[config.data.prompt_key].tolist()

        chat_lst = [chat.tolist() for chat in chat_lst]

        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

        total_samples = len(dataset)
        # real_batch_size = data.batch['input_ids'].shape[0]
        config_batch_size = config.data.batch_size
        dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
        num_batch = (total_samples // config_batch_size) + 1
        output_lst = []  # We'll reshape at the end

        for batch_idx in tqdm(range(num_batch), desc='Generating'):
            # print(f'[{batch_idx+1}/{num_batch}] Start to process.')
            batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
            
            # Repeat the batch n_samples times
            repeated_chat_lst = []
            for chat in batch_chat_lst:
                repeated_chat_lst.extend([chat] * config.data.n_samples)
            
            inputs = tokenizer.apply_chat_template(repeated_chat_lst,
                                                 add_generation_prompt=True,
                                                 padding=True,
                                                 truncation=True,
                                                 max_length=config.rollout.prompt_length,
                                                 return_tensors='pt',
                                                 return_dict=True,
                                                 tokenize=True)
            
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            position_ids = compute_position_id_with_mask(attention_mask)

            batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

            data = DataProto.from_dict(batch_dict)
            # === PATCH: fill to dp_size ===
            # data, real_batch_size = pad_to_world_size(data, dp_size)
            world_size = wg.world_size
            data, real_batch_size = pad_to_world_size(data,world_size )

            
            # real_batch_size = data.batch['input_ids'].shape[0]
            
            # if real_batch_size % dp_size != 0:
            #     dummy_data_size = dp_size - real_batch_size % dp_size
            #     # dummy_data = data[:dummy_data_size]
            #     dummy_input_ids = data.batch['input_ids'][:1].repeat(dummy_data_size, 1)  # copy the first sample
            #     dummy_attention_mask = data.batch['attention_mask'][:1].repeat(dummy_data_size, 1)
            #     dummy_position_ids = data.batch['position_ids'][:1].repeat(dummy_data_size, 1)
            #     dummy_data = DataProto.from_dict({
            #         'input_ids': dummy_input_ids,
            #         'attention_mask': dummy_attention_mask,
            #         'position_ids': dummy_position_ids
            #     })
            #     data = DataProto.concat([data, dummy_data])
            #     print(
            #         f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
                # )

            batch_size = data.batch['input_ids'].shape[0]
            assert batch_size % world_size == 0, f'batch_size {batch_size} is not divisible by world_size {world_size}'

            # print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
            
            # # Generate all samples at once
            # print(len(data.batch['input_ids']))
            output = wg.generate_sequences(data)
            # Remove dummy data
            output = output[:real_batch_size]
            output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                               skip_special_tokens=False)
            # Remove padding
            pad_token = tokenizer.pad_token
            output_text_unpad = []
            for text in output_text:
                output_text_unpad.append(text.replace(pad_token, ''))

            output_lst.extend(output_text_unpad)
            
            if (batch_idx+1) % 50 == 0:
                try:
                    print(f'[{batch_idx+1}/{num_batch}] Finished generating {len(output_lst)} samples.')
                    # Reshape output_lst from (total_samples,) to (n_data, n_samples)
                    total_samples = len(output_lst)
                    n_data = total_samples // config.data.n_samples
                    output_lst_to_save = np.array(output_lst).reshape(n_data, config.data.n_samples).tolist()
                    # Add to the data frame
                    dataset['responses'] = output_lst_to_save + [None] * (len(dataset) - len(output_lst_to_save))
                    del output_lst_to_save

                    # Write to a new parquet
                    output_dir = os.path.dirname(config.data.output_path)
                    makedirs(output_dir, exist_ok=True)
                    temp_path = os.path.join(output_dir, f'temp.parquet')
                    dataset.to_parquet(temp_path)
                except Exception as e:
                    print(e)
                    print('Error reshaping output_lst, skipping this step.')
                    # If reshaping fails, just continue with the current output_lst
                    pass

        # Reshape output_lst from (total_samples,) to (n_data, n_samples)
        total_samples = len(output_lst)
        n_data = total_samples // config.data.n_samples
        output_lst = np.array(output_lst).reshape(n_data, config.data.n_samples).tolist()

        # Add to the data frame
        dataset['responses'] = output_lst

        # Write to a new parquet
        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.to_parquet(config.data.output_path)
    
    output_dir = os.path.dirname(config.data.output_path)
    # Compute evaluation metrics
    prompts = dataset[config.data.prompt_key]
    responses = dataset['responses']  # Using the generated responses
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    passes = 0
    total = len(dataset)
    total_scores = []
    scores = []
    total_thinks = []
    thinks = []
    total_token_counts = []  # === ===
    
    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    for i in range(total):
        response_lst = responses[i]
        data_source = data_sources[i]
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        score_lst = []
        think_lst=[]
        token_lens = []  # ===  ===

        for r in response_lst:
            reward, score, think = reward_fn(r, ground_truth)
            score_lst.append(score)
            think_lst.append(think)
            tokens = tokenizer(r, return_tensors="pt")["input_ids"][0]
            token_lens.append(len(tokens))  # === ===

        scores.append(score_lst)
        thinks.append(think_lst)
        max_score = np.max(score_lst)
        total_scores.append(score_lst)
        total_thinks.append(think_lst)
        total_token_counts.append(np.mean(token_lens))  # === ===

        if max_score == 1:
            passes += 1
            
    # Save the scores to the dataset
    dataset['scores'] = scores
    dataset['thinks'] = thinks
    dataset['token_len'] = total_token_counts  # === ===
    print(f'Saving dataset with scores to {config.data.output_path}')
    dataset.to_parquet(config.data.output_path)

    n_samples = config.data.n_samples
    pass_at_n = passes / total
    pass_at_1 = np.mean(total_scores)
    avg_thinks = np.mean(total_thinks)
    avg_token_count_overall = np.mean(total_token_counts)  # === ===
    # Save metrics to CSV
    csv_path = os.path.join(output_dir, 'pass.csv')
    
    # Prepare the row data
    # Extract the dataset name from the path
    dataset_name = os.path.basename(config.data.path)
    row_data = {
        'model_path': config.model.path,
        'dataset': dataset_name,
        'pass@1': pass_at_1,
        f'pass@{n_samples}': pass_at_n,
        'token_len': avg_token_count_overall,  # === ===
        "avg_think_rate": avg_thinks
    }

    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]
    
    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

# Add the select_reward_fn from main_eval.py
def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.compute_score
    else:
        from deepscaler.rewards.math_reward import deepscaler_reward_fn
        return deepscaler_reward_fn

if __name__ == '__main__':
    main()
