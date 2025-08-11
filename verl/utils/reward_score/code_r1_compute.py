import re
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .code_r1_reward.firejail_exec import code_exec_firejail as code_executor
from .code_r1_reward.utils import BASE_IMPORTS, BASE_LEETCODE_IMPORTS
from .reward_config import ScoringConfig

import time

MAX_WORKERS = 2
MAX_TEST_CASES = 16
ERROR_MSG_PREFIX = "Execution failed: "
from .code_r1_reward.firejail_exec import _DEFAULT_TIMEOUT_SECONDS

scoring_config = ScoringConfig()
CODE_EXTRACTION_PATTERN = re.compile(r"</think>.*```python\n(.*?)\n```", re.DOTALL)

def extract_code(content):
    match = CODE_EXTRACTION_PATTERN.search(content)
    return match.group(1) if match else None

def execute_code(code, input_data, expected_output, timeout):
    success, output = code_executor(
        code=code, 
        stdin=input_data, 
        timeout=timeout
    )
    return success, output, input_data, expected_output

def sample_test_cases(inputs, outputs, max_samples):
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), min(len(inputs), max_samples))
    sampled_inputs = []
    sampled_outputs = []
    
    for idx in indices:
        in_data = inputs[idx]
        out_data = outputs[idx]
        
        if isinstance(in_data, list):
            in_data = ", ".join(map(str, in_data))
        if isinstance(out_data, list):
            out_data = ", ".join(map(str, out_data))
            
        sampled_inputs.append(in_data)
        sampled_outputs.append(out_data)
    
    return sampled_inputs, sampled_outputs

def _compute_score(solution, ground_truth, extra_info):
    log_messages = []
    solution_code = extract_code(solution)
    
    if not solution_code:
        log_messages.append("─" * 20 + " Invalid Code Format " + "─" * 20)
        log_messages.append(solution)
        return scoring_config.format_error_score, "\n".join(log_messages)

    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    timeout = ground_truth.get("timeout", _DEFAULT_TIMEOUT_SECONDS)
    max_test_cases = ground_truth.get("max_test_cases", MAX_TEST_CASES)
    log_messages.append("─" * 20 + " Valid Code Detected " + "─" * 20)

    start_time = time.time()
    
    # Handling functional test cases
    if "pytest" in ground_truth or "functional" in ground_truth:
        solution_code = "\n\n".join([BASE_IMPORTS, BASE_LEETCODE_IMPORTS, solution_code])
        
        if "functional" in ground_truth:
            test_cases = ground_truth["functional"]
            
            if isinstance(test_cases, str):
                all_result = []
                success, output = code_executor(
                    solution_code + "\n\n" + test_cases, 
                    timeout=timeout
                )
                if not success:
                    all_result.append(scoring_config.incorrect_score)
                else:
                    all_result.append(scoring_config.correct_score)

                    # return scoring_config.incorrect_score, "\n".join(log_messages)
                # return scoring_config.correct_score, "\n".join(log_messages)
            
            elif isinstance(test_cases, list):
                selected = random.sample(test_cases, min(max_test_cases, len(test_cases)))
                test_programs = [solution_code + "\n\n" + case for case in selected]
                
                with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(test_programs))) as executor:
                    futures = [executor.submit(code_executor, prog, None, timeout) for prog in test_programs]
                    all_result = []
                    for future in as_completed(futures):
                        success, output = future.result()
                        if not success:
                            all_result.append(scoring_config.incorrect_score)
                        else:
                            all_result.append(scoring_config.correct_score)

                            # for f in futures:
                            #     f.cancel()
                            # return scoring_config.incorrect_score, "\n".join(log_messages)
            else:
                raise ValueError("Unsupported functional test format")
        else:
            all_result = []
            success, output = code_executor(solution_code, pytest=ground_truth["pytest"])
            
            if not success:
                all_result.append(scoring_config.incorrect_score)
            else:
                all_result.append(scoring_config.correct_score)
            
        # return scoring_config.correct_score, "\n".join(log_messages)
    
    # Handle standard input/output tests
    elif "inputs" in ground_truth and "outputs" in ground_truth:
        inputs, outputs = sample_test_cases(
            ground_truth["inputs"], 
            ground_truth["outputs"], 
            max_test_cases
        )
        
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(inputs))) as executor:
            futures = [executor.submit(execute_code, solution_code, i, o, timeout) for i, o in zip(inputs, outputs)]
            all_result = []
            for future in as_completed(futures):
                success, output, stdin, stdout = future.result()
                if not success or output.strip() != stdout.strip():
                    all_result.append(scoring_config.incorrect_score)
                else:
                    all_result.append(scoring_config.correct_score)
    
    # 处理文件输入/输出测试
    elif "inputs_file" in ground_truth and "outputs_file" in ground_truth:
        with open(ground_truth["inputs_file"]) as f:
            inputs = [f.read()]
        with open(ground_truth["outputs_file"]) as f:
            outputs = [f.read()]
        
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(inputs))) as executor:
            futures = [executor.submit(execute_code, solution_code, i, o, timeout) for i, o in zip(inputs, outputs)]
            for future in as_completed(futures):
                success, output, stdin, stdout = future.result()
                all_result = []
                if not success or output.strip() != stdout.strip():
                    all_result.append(scoring_config.incorrect_score)
                else:
                    all_result.append(scoring_config.correct_score)
    
    else:
        raise ValueError(f"Unsupported test format: {ground_truth.keys()}")
    
    return sum(all_result) * 1.0 / len(all_result), "\n".join(log_messages)

def compute_score(solution, ground_truth, extra_info=None, debug=False, is_longcot: bool = False):

    if is_longcot:
        if ("</think>" not in solution):
            reward = scoring_config.wo_eos_think
            return reward


        if ("<think>" not in solution):
            reward = scoring_config.wo_bos_think
            return reward


    if extra_info is None:
        extra_info = []
    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()
    
    score, log = _compute_score(
        solution=solution,
        ground_truth=ground_truth,
        extra_info=extra_info
    )
    
    return score