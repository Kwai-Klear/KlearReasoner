# ✨ Klear-Reasoner: Advancing Reasoning Capability via Gradient-Preserving Clipping Policy Optimization
We present Klear-Reasoner, a model with long reasoning capabilities that demonstrates careful deliberation during problem solving, achieving outstanding performance across multiple benchmarks. We investigate two key issues with current clipping mechanisms in RL: Clipping suppresses critical exploration signals and ignores suboptimal trajectories. To address these challenges, we propose **G**radient-**P**reserving clipping **P**olicy **O**ptimization (**GPPO**) that gently backpropagates gradients from clipped tokens.  

---

<!-- ## 🚀 Quick Links
| Resource | Link |
|---|---|
| 🤗 Model Hub | [Klear-Reasoner-8B](https://huggingface.co/klear-team/klear-reasoner-8b) |
| 📄 Technical Report | [arXiv:250x.xxxxx](https://arxiv.org/abs/250x.xxxxx) |
| 🐛 Issues & Discussions | [GitHub Issues](https://github.com/klear-team/klear-reasoner/issues) |
| 📧 Contact | klear-reasoner@kuaishou.com |

--- -->

| Resource | Link |
|---|---|
| 🤗 Model Hub | [Klear-Reasoner-8B](https://huggingface.co/Suu/Klear-Reasoner-8B) |
| 🐛 Issues & Discussions | [GitHub Issues](https://github.com/suu990901/KlearReasoner/issues) |
| 📧 Contact | suzhenpeng13@163.com |


## 📌 Overview

<div align="center">
<img src="./docker/main_result.png" width="100%"/>

<sub>Benchmark accuracy of Klear-Reasoner-8B on AIME 2024/2025 (avg@64), LiveCodeBench V5 (2024/08/01-2025/02/01, avg@8), and v6 (2025/02/01-2025/05/01, avg@8).</sub>
</div>

Klear-Reasoner is an 8-billion-parameter reasoning model that achieves **SOTA** performance on challenging **math and coding benchmarks**:

| Benchmark | AIME 2024 | AIME 2025 | LiveCodeBench V5 | LiveCodeBench V6 |
|---|---|---|---|---|
| **Score** | **90.5 %** | **83.2 %** | **66.0 %** | **58.1 %** |

The model combines:
1. **Quality-centric long CoT SFT** – distilled from DeepSeek-R1-0528.
2. **Gradient-Preserving Clipping Policy Optimization (GPPO)** – a novel RL method that **keeps gradients from clipped tokens** to boost exploration & convergence.

---

<!-- ## 🛠️ Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
``` -->

<!-- ### 2. Load Model & Generate
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "klear-team/klear-reasoner-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "Prove that for all positive integers n, n^3 + 2n is divisible by 3."
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=8192,
    temperature=0.6,
    top_p=0.95,
    do_sample=True
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

--- -->


## 📐 GPPO (Gradient-Preserving Clipping Policy Optimization)

GPPO is a **plug-and-play** replacement for PPO/GRPO that keeps the clipped tokens **in the computational graph** and lets their gradients flow in a **bounded, controlled** way.


### Problem with Vanilla Clipping  
Classic importance-ratio clipping (PPO/GRPO) drops all tokens whose ratio  
$r_t^{(j)}=\pi_\theta/\pi_{\text{old}}$ falls outside $[1-\varepsilon_l,\ 1+\varepsilon_h]$.  
Two side-effects appear:

- **High-entropy exploratory tokens** (large $r$, positive advantage) are killed → less exploration.  
- **Negative trajectories** (small $r$, negative advantage) are ignored → slower correction.


### GPPO Surrogate Loss (Token-Level GRPO)  

Let  
- $\delta = r_t^{(j)}(\theta)=\pi_\theta/\pi_{\text{old}}$ (importance ratio)  
- $\tilde A^{(j)}$ = group-relative advantage  
- $\text{sg}(\cdot)$ = stop-gradient (detach from back-prop)

The **GPPO objective** is  


![GPPO Loss](https://github.com/suu990901/KlearReasoner/blob/main/docker/CodeCogsEqn.svg)


- **Forward**: behaves exactly like Clip-Higher.  
- **Backward**: the fraction $\frac{1\pm\varepsilon}{\text{sg}(\delta)}$ keeps the clipped magnitude **but still propagates** a mild gradient.


### Gradient Expression  

Let $\phi_\theta(a_{j,t},s_{j,t})$ be the policy-gradient vector.  
The **per-token gradient** is  

![gard](https://github.com/suu990901/KlearReasoner/blob/main/docker/CodeCogsEqn_1.svg)


where  

![condtion](https://github.com/suu990901/KlearReasoner/blob/main/docker/CodeCogsEqn_2.svg)


- **Bounded** gradients avoid explosion.  
- **Never zero** → every token contributes to learning.


### General Form with Tunable Scaling ($\beta_1$, $\beta_2$)  

For finer-grained control:  

![general_loss](https://github.com/suu990901/KlearReasoner/blob/main/docker/CodeCogsEqn_3.svg)

Empirically we set $\beta_1 = \beta_2 = 1$.

---

## 📊 Benchmark Results (Pass@1)

| Model | AIME2024<br>avg@64 | AIME2025<br>avg@64 | HMMT2025<br>avg@64 | LCB V5<br>avg@8 | LCB V6<br>avg@8 |
|-------|--------------------|--------------------|--------------------|-----------------|-----------------|
| AReal-boba-RL-7B | 61.9 | 48.3 | 29.4 | 34.3 | 31.0† |
| MiMo-7B-RL | 68.2 | 55.4 | 35.7 | 57.8 | 49.3 |
| Skywork-OR1-7B | 70.2 | 54.6 | 35.7 | 47.6 | 42.7 |
| AceReason-Nemotron-1.1-7B | 72.6 | 64.8 | 42.9 | 57.2 | 52.1 |
| POLARIS-4B-Preview  | 81.2 | _79.4_ | 58.7 | 58.5† | 53.0† |
| Qwen3-8B | 76.0 | 67.3 | 44.7† | 57.5 | 48.4† |
| Deepseek-R1-0528-Distill-8B  | _86.0_ | 76.3 | 61.5 | 61.0† | 51.6† |
| OpenReasoning-Nemotron-7B  | 84.7 | 78.2 | 63.5 | _65.6_† | _56.3_† |
| Klear-Reasoner-8B-SFT | 75.6 | 70.1 | 57.6 | 58.5 | 49.6 |
| Klear-Reasoner-8B | 83.2 | 75.6 | 60.3 | 61.6 | 53.1 |
| *w/ 64K Inference Budget*  | **90.5** | **83.2** | **70.8** | **66.0** | **58.1** |

> We report the average `pass@1` results (avg@_n_), with all other evaluation metrics following the DeepSeek-R1 assessment framework (temperature=0.6, top_p=0.95).  


---

## 🧪 Training
### Configure the experimental environment
```bash
git clone https://github.com/suu990901/Klear_Reasoner.git
cd Klear_Reasoner
pip install -r requirements.txt
```
For the code, we use [Firejail](https://github.com/netblue30/firejail) for the **sandbox** environment. Additionally, we implemented multi-process control based on [Pebble](https://github.com/noxdafox/pebble), which allows us to reclaim all resources allocated to a task when execution times out. For mathematics, we use [math_verify](https://github.com/huggingface/Math-Verify) for judging.

### Using Ray for Multi-Node Training
For multi-node training​​, ensure ​​all nodes are started and connected via Ray​​ before executing the training script. Below is a brief setup guide for Ray across multiple machines:
#### Step 1: Start Ray on the Head Node (node0)

On the first node (typically called `node0`), run:

```bash
ray start --head --dashboard-host=0.0.0.0
```

Get the IP address of the master node.
```bash
MASTER_IP=$(hostname -I | awk '{print $1}')
```
#### Step 2: Connect Other Nodes (e.g., node1)

On each additional worker node (e.g., `node1`), run the following, replacing the IP with that of your head node:

```bash
ray start --address=\"$MASTER_IP:6379\"
```

### RL Training
Run the following script on the master node to start the training task.

```bash
bash recipe/dapo/perf_run_dapo_ours_math.sh # For Math RL
bash recipe/dapo/perf_run_dapo_ours_code.sh # For Code RL
```

In the startup script, you need to set the following variables:
```bash
YOUR_MODEL_PATH="<your_model_path>"
CKPTS_SAVE_DIR="<ckpts_save_path>"
YOUR_TRAIN_FILE="<train_data_path>"
YOUR_TEST_FILE="<test_data_path>"
```

---

<!-- ## 🔍 Key Techniques
| Component | Description |
|---|---|
| **GPPO** | Gradient-Preserving Clipping Policy Optimization. Retains clipped-token gradients; stabilizes training while boosting exploration. |
| **Soft Reward** | For code tasks, reward = (passed tests / total tests) instead of binary 0/1. |
| **Zero-Advantage Filtering** | Removes prompt groups whose advantages are all zero → clearer gradients. |
| **YaRN Extension** | Extends context to 64 K tokens during inference for better long-CoT reasoning. | -->


<!-- --- -->

<!-- ## 🤝 Citation
```bibtex
@misc{klear2025reasoner,
  title={Klear-Reasoner: Advancing Reasoning Capability via Gradient-Preserving Clipping Policy Optimization},
  author={Su, Zhenpeng and Pan, Leiyu and Bai, Xue and Liu, Dening and Dong, Guanting and Huang, Jiaming and Hu, Wenping and Zhang, Fuzheng and Zhou, Guorui},
  year={2025},
  eprint={250x.xxxxx},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
``` -->

<!-- --- -->

