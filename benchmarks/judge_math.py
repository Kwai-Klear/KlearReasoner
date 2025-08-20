import sys
import json
from math_verify import parse, verify
from tqdm import tqdm
from collections import Counter


def remove_think(content):
    return content.split('</think>')[-1].strip()

def math_verify_from_sky(solution_str: str, ground_truth: str):
    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    
    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return -1.0
    
    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return -1.0
    
    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return 1.0
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue
    
    # Very unlikely to be correct after the above matches
    return -1.0

def main(data_path):
    result_data_list = []
    with open(data_path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            try:
                response = remove_think(data["messages"][-1]["content"])
                answer = data["meta"]["meta"]["solution"].strip()

                if isinstance(response, list):
                    if not warned:
                        print("Warning: response is a list, using the first element", file=sys.stderr)
                        warned = True
                    response = response[0]
                if isinstance(answer, list):
                    if not warned:
                        print("Warning: answer is a list, using the first element", file=sys.stderr)
                        warned = True
                    answer = answer[0]

                if isinstance(answer, list):
                    assert len(answer) == 1, f"Expected 1 answer, got {len(answer)}"
                    answer = answer[0]
                if isinstance(response, list):
                    assert len(response) == 1, f"Expected 1 response, got {len(response)}"
                    response = response[0]
                
                data["is_correct"] = math_verify_from_sky(response, answer)
            except Exception as e:
                data["is_correct"] = -1.0
            result_data_list.append(data)

    for idx, data in tqdm(enumerate(result_data_list)):
        if data["is_correct"] != 1:
            data["id"] = idx % 30
            data["meta"]["messages"][-1]["content"] = remove_think(data["meta"]["messages"][-1]["content"])
            data["messages"][-1]["content"] = remove_think(data["messages"][-1]["content"])
            # print(json.dumps(data, ensure_ascii=False))
    counter = Counter()

    for idx, data in tqdm(enumerate(result_data_list)):
        if data["is_correct"] != 1:
            data["id"] = idx % 30
            counter[data["id"]] += 1  

    print(f"Pass rate: {(len(result_data_list)-sum(counter.values())) / len(result_data_list):.2%}")



if __name__ == "__main__":
    data_path = sys.argv[1]
    main(data_path)