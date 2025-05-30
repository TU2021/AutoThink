
# 🧠 AutoThink: Adaptive Reasoning in R1-Style Models


<p align="center">
          🔗 <a href="https://github.com/ScienceOne-AI/AutoThink">Codebase</a>&nbsp&nbsp | 🤗 <a href="https://huggingface.co/collections/SONGJUNTU/autothink-682624e1466651b08055b479">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2505.10832">Paper</a>&nbsp&nbsp | &nbsp&nbsp📖 <a href="https://mp.weixin.qq.com/s/qcGrNjIqU1cLSg_31wijJg"> WeChat Chinese Version</a>&nbsp&nbsp
</p>



**AutoThink** is a reinforcement learning framework designed to equip R1-style language models with **adaptive reasoning** capabilities. Instead of always thinking or never thinking, the model learns **when** to engage in explicit reasoning, balancing performance and efficiency.

This repository implements **AutoThink**, as described in our paper:

> *Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL*  

![framework1](./assets/1.png)
---

## 📰 News
- ***[2025/05/28]*** Our work was featured on the **QbitAI** WeChat public account: 📖 [Chinese Version](https://mp.weixin.qq.com/s/qcGrNjIqU1cLSg_31wijJg)

- ***[2025/05/27]*** We apply *AutoThink* to the SOTA 7B model [Skywork-OR1-Math-7B](https://huggingface.co/Skywork/Skywork-OR1-Math-7B). *AutoThink* reduces reasoning token usage by **56%** with **less than 2% accuracy degradation**.  We also updated the paper to fix minor issues and released the corresponding trained model.  

- ***[2025/05/16]*** We release the [Code](https://github.com/ScienceOne-AI/AutoThink), [Models](https://huggingface.co/collections/SONGJUNTU/autothink-682624e1466651b08055b479), and [Paper](https://arxiv.org/abs/2505.10832) for *AutoThink*.  


## 🚀 Features

- 🧩 **Minimal Prompting** with ellipsis (`<think>\n...\n`) to activate stochastic thinking.
- 🎯 **Multi-stage RL** to stabilize, reinforce, and prune reasoning behavior.
- ⚙️ Integrated with the [`verl`](https://github.com/volcengine/verl) framework.
- 📊 Benchmarked on five mathematical reasoning datasets: MATH, Minerva, Olympiad, AIME24, AMC23.

![framework2](./assets/2.png)

---

## 📦 Installation

Please clone the official [DeepScaleR](https://github.com/agentica-project/rllm) repository and follow its setup instructions:

Then, **replace the following three folders** in the original repo with ours:

```bash
cp -r code-release/verl     deepscaler/
cp -r code-release/scripts  deepscaler/
cp -r code-release/deepscaler deepscaler/
```

Install the environment:
```bash
# Recommend Python 3.10.
cd deepscaler
pip install -e ./verl
pip install -e .
```

The raw training data is located in `deepscaler/data/[train|test]`, along with preprocessing scripts. To convert the raw data into Parquet files for training, run:

```bash
# Output parquet files in data/*.parquet.
python scripts/data/deepscaler_dataset.py
```

---

## 💡 Different Prompt Strategies


You can control the model's reasoning behavior by modifying the `chat_template` field in `tokenizer_config.json`. Update the value with one of the following:

- **Standard Prompt** (default for Distill-R1, no changes needed):

```json
"<|Assistant|><think>\n"
```

- **No-Thinking Prompt** (forces minimal reasoning):

```json
"<|Assistant|><think>\nOkay, I think I have finished thinking.\n</think>\n\n"
```


- **Ellipsis Prompt** (adaptive reasoning mode):

```json
"<|Assistant|><think>\n...\n"
```


These prompts enable different reasoning behaviors.  
Before AutoThink training, please replace the default `chat_template` with **Ellipsis Prompt** and keep the inference prompt consistent.



## 🏋️ Training

AutoThink training proceeds in **three stages** with different reward designs:

```bash
# Stage 1: Stabilize dual-mode reasoning
bash scripts/train_stage1.sh

# Stage 2: Reinforce accurate behavior
bash scripts/train_stage2.sh

# Stage 3: Prune redundant reasoning
bash scripts/train_stage3.sh
```

Make sure to configure your model paths and data in `scripts/train_*.sh`.


---

## 📈 Evaluation

After training, evaluate the model using:

```bash
bash scripts/eval/eval_model_1.5b.sh
```


---


## 📊 Results

AutoThink achievesefficiency–accuracy trade-offs, and exhibits two inference modes:

![results](./assets/3.png)
![results](./assets/5.png)
![modes](./assets/4.png)
---

## 📄 Citation

```bibtex
@article{tu2025learning,
  title={Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL},
  author={Tu, Songjun and Lin, Jiahao and Zhang, Qichao and Tian, Xiangyu and Li, Linjing and Lan, Xiangyuan and Zhao, Dongbin},
  journal={arXiv preprint arXiv:2505.10832},
  year={2025}
}
```

---

## 🔍 Acknowledgements

We build and reference on the following open source trunks, and thank the following sources for their contributions to the LLM-Reasoning open source community:
- [verl](https://github.com/volcengine/verl)
- [DeepScaleR](https://github.com/agentica-project/rllm)
- [ThinkPrune](https://github.com/UCSB-NLP-Chang/ThinkPrune)
