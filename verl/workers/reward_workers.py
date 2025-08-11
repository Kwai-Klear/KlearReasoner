
"""
The worker-like implement for reward compute
"""

import logging
import os
import warnings
import psutil
import wandb
import datetime
import time

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch


class RewardWorker(Worker):
    def __init__(self, config, reward_fn, val_reward_fn):
        super().__init__()
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_reward(self, data: DataProto):
        with self.timing_record('compute_reward/compute_reward'):
            res_dict = self.reward_fn(data, True)
            res = DataProto.from_dict({'reward_tensor': res_dict['reward_tensor']},
                                      non_tensors=res_dict['reward_extra_info'])
        return res

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_val_reward(self, data: DataProto):
        with self.timing_record('compute_reward/compute_val_reward'):
            res_dict = self.val_reward_fn(data, True)
            #print(f"extra_info = {res_dict['reward_extra_info']}", flush=True)
            res = DataProto.from_dict({'reward_tensor': res_dict['reward_tensor']},
                                      non_tensors=res_dict['reward_extra_info'])
        return res
