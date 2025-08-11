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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score, dapo_wi_rllm_compute_score
from verl.utils.reward_score.reward_config import ScoringConfig
import torch
from collections import defaultdict
import multiprocessing
from functools import partial
import time
import re
import gc
import logging
from pebble import ProcessPool, ProcessExpired  # Using pebble instead of concurrent.futures
from concurrent.futures import TimeoutError

# Configuring Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

scoring_config = ScoringConfig()

class DAPORewardManager:
    """Improved reward manager to resolve timeout and efficiency issues."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        overlong_filter=False
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or dapo_wi_rllm_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.overlong_filter = overlong_filter
        
        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def _prepare_item_data(self, data_item):
        """Prepare processing data for a single data item, reducing serialization overhead."""
        prompt_ids = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        
        # Remove the eos token at the end.
        if response_str.endswith(self.tokenizer.eos_token):
            response_str = response_str[:-len(self.tokenizer.eos_token)]
        
        # Determine whether the answer generation is complete.
        is_finish = None
        if self.overlong_filter:
            if self.tokenizer.eos_token_id == response_ids[-1] or self.tokenizer.pad_token_id == response_ids[-1]:
                is_finish = 1
            else:
                is_finish = 0
        
        return {
            'prompt_str': prompt_str,
            'response_str': response_str,
            'ground_truth': data_item.non_tensor_batch['reward_model']['ground_truth'],
            'data_source': data_item.non_tensor_batch[self.reward_fn_key],
            'extra_info': data_item.non_tensor_batch.get('extra_info', None),
            'valid_response_length': valid_response_length,
            'is_finish': is_finish
        }

    @staticmethod
    def _compute_single_score(item_data, compute_score_func):
        """Calculate the score of a single data item (static method, to reduce serialization issues)"""
        try:
            result = compute_score_func(
                data_source=item_data['data_source'],
                solution_str=item_data['response_str'],
                ground_truth=item_data['ground_truth'],
                extra_info=item_data['extra_info'],
            )
            return {
                'index': item_data['index'],
                'result': result,
                'valid_response_length': item_data['valid_response_length'],
                'response_str': item_data['response_str'],
                'is_finish': item_data['is_finish']
            }
        except Exception as e:
            print(f"Error processing item {item_data['index']}: {str(e)}")
            return {
                'index': item_data['index'],
                'result': -1.0,  # 默认值
                'valid_response_length': item_data['valid_response_length'],
                'response_str': item_data['response_str'],
                'is_finish': item_data['is_finish']
            }
    
    def _create_default_result(self, item_data):
        """Create default results for timeouts or exceptions"""
        return {
            'index': item_data['index'],
            'result': -1.0,
            'valid_response_length': item_data['valid_response_length'],
            'response_str': item_data['response_str'],
            'is_finish': item_data['is_finish']
        }

    def __call__(self, data: DataProto, return_dict: bool = False):
        if 'rm_scores' in data.batch.keys():
            return {"reward_tensor": data.batch['rm_scores']} if return_dict else data.batch['rm_scores']

        # Prepare processing data for all data items
        items_data = []
        for idx in range(len(data)):
            item_data = self._prepare_item_data(data[idx])
            item_data['index'] = idx  # Add an index for sorting results
            items_data.append(item_data)
        
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        results = [None] * len(items_data)  # Preallocate result list
        
        # Using a process pool with timeout control
        TIMEOUT_PER_TASK = 500  # 500 seconds timeout for each task
        MAX_WORKERS =  10 
        
        # Use Pebble's ProcessPool instead of ProcessPoolExecutor
        with ProcessPool(max_workers=MAX_WORKERS) as pool:
            # Submit all tasks
            compute_func = partial(self._compute_single_score, compute_score_func=self.compute_score)
            futures = [pool.schedule(compute_func, args=(item,), timeout=TIMEOUT_PER_TASK) for item in items_data]
            future_to_item = {fut: items_data[i] for i, fut in enumerate(futures)}
            
            # Processing task results
            start_time = time.time()
            for future in futures:
                item = future_to_item[future]
                try:
                    result = future.result()  # Waiting for the result will throw a timeout exception.
                    results[item['index']] = result
                    logger.info(f"Processed item {item['index']+1}/{len(items_data)}")
                except TimeoutError as error:
                    logger.error(f"Task for item {item['index']} timed out: {error}")
                    results[item['index']] = self._create_default_result(item)
                except ProcessExpired as error:
                    logger.error(f"Task for item {item['index']} expired: {error}")
                    results[item['index']] = self._create_default_result(item)
                except Exception as error:
                    logger.error(f"Error in task for item {item['index']}: {error}")
                    results[item['index']] = self._create_default_result(item)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Completed all tasks in {elapsed_time:.2f}s")

        # Processing results (in original index order)
        for result in results:
            if result is None:  # Dealing with unfinished tasks
                continue
                
            idx = result['index']
            score_result = result['result']
            valid_response_length = result['valid_response_length']
            response_str = result['response_str']
            is_finish = result['is_finish']
            
            # Processing score results
            score = score_result["score"] if isinstance(score_result, dict) else score_result
            reward_extra_info["acc"].append(int(score == scoring_config.correct_score))

            if self.overlong_filter and is_finish is not None:
                reward_extra_info["is_finish"].append(is_finish)
            
            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                score += overlong_reward
                
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)
            
            reward_tensor[idx, valid_response_length - 1] = score
        
        # Cleaning up resources
        gc.collect()
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor