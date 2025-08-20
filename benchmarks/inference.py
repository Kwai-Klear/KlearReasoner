# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser
import json
import os
import time
from tqdm import tqdm
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("v")

def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine parameters
    # parser.set_defaults(model="./KlearReasoner-8B")
    # Add engine initialization parameters
    engine_group = parser.add_argument_group("Engine Initialization Parameters")
    engine_group.add_argument("--model", type=str,
                            default="./KlearReasoner-8B",
                            help="Model path or identifier")
    engine_group.add_argument("--max-model-len", type=int, default=81920, 
                            help="Maximum model length (in tokens)")
    engine_group.add_argument("--rope-scaling", type=str, 
                            default='{"rope_type":"yarn","factor":2.5,"original_max_position_embeddings":32768}', 
                            help="RoPE scaling type, such as 'linear', 'dynamic', etc.")
    engine_group.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                            help="Target GPU memory utilization (0.0 to 1.0)")
    engine_group.add_argument("--tensor-parallel-size", type=int, default=8,
                            help="Tensor parallelism degree")
    engine_group.add_argument("--max-num-seqs", type=int, default=128,
                            help="Maximum number of sequences")
    engine_group.add_argument("--seed", type=int, default=0,
                            help="Random seed")

    # Add sampling parameters
    sampling_group = parser.add_argument_group("Sampling Parameters")
    sampling_group.add_argument("--max-tokens", type=int, default=65536)
    sampling_group.add_argument("--temperature", type=float, default=0.6)
    sampling_group.add_argument("--top-p", type=float, default=0.95)
    sampling_group.add_argument("--top-k", type=int, default=20)
    sampling_group.add_argument("--n", type=int, default=64)
    
    # Add dataset and output parameters
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument("--dataset-path", type=str, required=True, help="Dataset file path")
    data_group.add_argument("--output-data", type=str, default="./output", help="Output data save directory")
    data_group.add_argument("--cache-dir", type=str, default="./cache", help="Model cache directory for warm start")
    data_group.add_argument("--batch-size", type=int, default=128)

    return parser


def main(args: dict):
    # Load dataset
    data_path = args.pop("dataset_path")
    output_dir = args.pop("output_data")
    batch_size = args.pop("batch_size")
    data_list = load_dataset(data_path)
    n = args.pop("n")

    # Repeat dataset n times
    original_len = len(data_list)
    data_list = data_list * n
    logger.info(f"Dataset repeated {n} times, expanded from {original_len} entries to {len(data_list)} entries")

    # Ensure output and cache directories exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract sampling parameters
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")


    # Parse rope_scaling from string to dict if needed
    rope_scaling = args.get("rope_scaling")
    if isinstance(rope_scaling, str):
        try:
            rope_scaling = json.loads(rope_scaling)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing rope_scaling JSON: {e}")
            rope_scaling = {"rope_type": "yarn", "factor": 2.5, "original_max_position_embeddings": 32768}

    # Print engine information
    logger.info("=" * 50)
    logger.info("Initializing vLLM Engine")
    logger.info(f"Model: {args.get('model', 'unknown')}")
    logger.info(f"Max Model Length: {args.get('max_model_len', 'default')}")
    logger.info(f"RoPE Scaling: {rope_scaling}")
    logger.info(f"Tensor Parallel Size: {args.get('tensor_parallel_size', 1)}")
    logger.info(f"GPU Memory Utilization: {args.get('gpu_memory_utilization', 0.9)}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info("=" * 50)

    # Create LLM instance using parsed arguments
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    
    # Prepare hf_overrides with rope_scaling and max_model_len
    hf_overrides = {
        "rope_scaling": rope_scaling,
        "max_model_len": args.get("max_model_len", 81920)
    }
    
    llm = LLM(
        model=args.get("model"),
        tensor_parallel_size=args.get("tensor_parallel_size", 8),
        gpu_memory_utilization=args.get("gpu_memory_utilization", 0.9),
        enable_prefix_caching=True,
        max_num_seqs=args.get("max_num_seqs", 128),
        hf_overrides=hf_overrides,
        seed=args.get("seed", 0),
        repetition_penalty=1.05
    )
  
    logger.info("vLLM engine initialization completed")

    # Create sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=args.get("seed", 0)
    )
    print(sampling_params)

    # Process data and write results in real-time
    process_data_batch(llm, data_list, sampling_params, output_dir, batch_size)


def process_data_batch(llm, data_list, sampling_params, output_dir, batch_size):
    """Process data using batch processing and write results in real-time"""
    results_file = os.path.join(output_dir, "inference_results.jsonl")
    
    # Track number of processed items
    processed_count = 0
    
    # Check for partial results
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            processed_count = sum(1 for _ in f)
        logger.info(f"Found existing results file, {processed_count} entries already processed")
        
    # Continue from last processed position
    remaining_data = data_list[processed_count:]
    if not remaining_data:
        logger.info("All data processing completed")
        return
        
    logger.info(f"Starting to process remaining {len(remaining_data)} entries")
    
    # Process in batches
    with open(results_file, 'a', encoding='utf-8') as f:
        # Split data into batches
        for batch_start in range(0, len(remaining_data), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_data))
            batch = remaining_data[batch_start:batch_end]
            
            try:
                # Prepare batch messages list and original data
                batch_messages = []
                original_items = []
                
                for item in batch:
                    # Save original data item
                    original_items.append(item)
                    
                    # Directly use messages field from data
                    if "messages" in item:
                        batch_messages.append(item["messages"])
                    else:
                        logger.warning(f"Data item missing 'messages' field: {item}")
                        batch_messages.append([{"role": "user", "content": "Invalid data"}])
                
                # Batch call chat API
                logger.debug(f"Starting batch processing {batch_start} to {batch_end-1}")
                outputs = llm.chat(batch_messages, sampling_params, use_tqdm=True)
                
                # Process each output and save results
                for i, output in enumerate(outputs):
                    item_idx = processed_count + batch_start + i
                    original_item = original_items[i]
                    
                    # Get AI response
                    ai_response = output.outputs[0].text
                    
                    # Build complete messages list including model's response
                    complete_messages = batch_messages[i].copy()
                    complete_messages.append({"role": "assistant", "content": ai_response})
                    
                    # Build metadata
                    meta = original_item
                        
                    # Build complete result
                    result = {
                        "id": item_idx,
                        "meta": meta,
                        "messages": complete_messages
                    }
                    
                    # Write result and flush immediately
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()
                
                # Update progress
                logger.info(f"Processed {processed_count + batch_end}/{len(data_list)} entries")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_start}-{batch_end-1}: {str(e)}")
                # Log error but continue with next batch
                for i in range(len(batch)):
                    item_idx = processed_count + batch_start + i
                    original_item = batch[i]
                    
                    # Build metadata
                    meta = original_item
                    
                    error_result = {
                        "id": item_idx,
                        "meta": meta,
                        "messages": original_item.get("messages", []),
                        "error": str(e)
                    }
                    f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                    f.flush()
    
    logger.info(f"Data processing completed, results saved to {results_file}")


def load_dataset(dataset_path: str) -> list:
    """Load dataset"""
    data_list = []
    logger.info(f"Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading data"):
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Unable to parse JSON line: {line}")
    
    logger.info(f"Loaded {len(data_list)} entries")
    return data_list


if __name__ == "__main__":
    parser = create_parser()
    args = vars(parser.parse_args())
    main(args)
