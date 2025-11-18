# Sotopia-RL 完整强化学习训练流程详解

## 📋 目录

1. [整体架构概览](#整体架构概览)
2. [阶段 0: 数据收集](#阶段-0-数据收集)
3. [阶段 1: SFT 训练（行为克隆）](#阶段-1-sft-训练行为克隆)
4. [阶段 2: Reward Model 训练](#阶段-2-reward-model-训练)
5. [阶段 3: GRPO 强化学习训练](#阶段-3-grpo-强化学习训练)
6. [完整对话示例：Reward 计算全流程](#完整对话示例reward-计算全流程)
7. [关键代码解析](#关键代码解析)
8. [总结](#总结)

---

## 整体架构概览

### 完整 Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  阶段 0: 数据收集                                                │
│  使用 GPT-4 在 Sotopia 环境中进行 self-play 对话               │
│  → 得到完整的对话 episodes                                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 1: SFT 训练（Supervised Fine-Tuning）                     │
│  使用 GPT-4 对话数据训练 Qwen2.5-7B 模型                       │
│  → 得到 SFT Model（初始的对话 agent）                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 1.5: LLM Attribution 标注                                 │
│  使用 GPT-4 给每句话打 attribution 分数                        │
│  → 得到每句话的 reward 标签                                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 2: Reward Model 训练                                      │
│  使用 attribution 数据训练 Reward Model                        │
│  → 得到 Reward Model（能给每句话打分）                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 3: GRPO 强化学习训练                                      │
│  使用 Reward Model 指导 SFT Model 进行 RL 优化                 │
│  → 得到最终的 RL Agent（社交对话专家）                         │
└─────────────────────────────────────────────────────────────────┘
```

### 核心思想

**Sotopia-RL 的独特之处：**
- ✅ **Utterance-level Reward**: 每句话都有独立的 reward，而不是整个对话一个分数
- ✅ **Attribution-based**: 通过 LLM 评估每句话对目标的贡献度
- ✅ **Multi-dimensional**: 从多个社交维度评估（目标、关系、可信度等）
- ✅ **Single-turn Online RL**: 每次只生成一句话，立即得到 reward 反馈

---

## 阶段 0: 数据收集

### 目标

收集高质量的社交对话数据，作为后续训练的基础。

### 数据来源

使用 **GPT-4** 在 **Sotopia** 环境中进行 self-play 对话。

**Sotopia 环境：**
- 社交场景模拟平台
- 定义了各种社交场景（谈判、安慰、说服等）
- 为每个 agent 分配目标、背景、性格特征

### 数据收集流程

```
1. 从 Sotopia 数据库中选择场景
   ↓
2. 为两个 agent 分配角色、目标、背景
   ↓
3. 使用 GPT-4 作为两个 agent 进行对话
   ↓
4. 记录完整的对话历史
   ↓
5. 保存到 Redis 数据库
   ↓
6. 导出为 JSONL 格式
```

### 原始数据格式

**文件：** `sotopia_pi_episodes.jsonl`

```json
{
    "episode_id": "01J1234567890ABCDEFG",
    "scenario": "Two friends arguing about character costume",
    "agents": ["Mia Davis", "Benjamin Jackson"],
    "agents_background": {
        "Mia Davis": "50-year-old high school principal...",
        "Benjamin Jackson": "24-year-old environmental activist..."
    },
    "social_goals": {
        "Mia Davis": "Negotiate to dress up as the character...",
        "Benjamin Jackson": "Also want to dress up as the same character..."
    },
    "social_interactions": "Turn #0: Mia Davis said: \"Hello Benjamin...\"\nTurn #0: Benjamin Jackson said: \"Hello Mia...\"",
    "experiment_model_name_pairs": ["gpt-4", "gpt-4"],
    "rewards": [
        ["Mia Davis", {"believability": 7, "relationship": 8, "knowledge": 5, "goal": 9, ...}],
        ["Benjamin Jackson", {"believability": 8, "relationship": 9, "knowledge": 6, "goal": 7, ...}]
    ],
    "scores": {
        "Mia Davis": 9.0,
        "Benjamin Jackson": 7.0
    }
}
```

### 数据过滤

**脚本：** `scripts/annotate/process_sotopia_pi.py`

只保留 GPT-4 vs GPT-4 的高质量对话：

```python
# 过滤条件
for episode in episodes:
    if episode["experiment_model_name_pairs"] == ["gpt-4", "gpt-4"]:
        behavior_cloning_episodes.append(episode)
```

**输出：** `sotopia_pi_bc_episodes.jsonl`（behavior cloning episodes）

---

## 阶段 1: SFT 训练（行为克隆）

### 目标

训练一个初始的对话模型，让它学会像 GPT-4 一样进行社交对话。

### 为什么需要 SFT？

- ❌ 直接用随机初始化的模型做 RL 很难收敛
- ✅ 先用 GPT-4 数据做监督学习，给模型一个好的起点
- ✅ 相当于"模仿学习"，学习专家的对话策略

### 数据准备

**输入：** `sotopia_pi_bc_episodes.jsonl`（GPT-4 对话数据）

**处理脚本：** `scripts/data_process/generate_sft_from_episodes.py`

**处理流程：**

```python
# 1. 遍历每个 episode
for episode in episodes:
    # 2. 解析对话历史
    conversation = parse_conversation(episode)
    # conversation = [
    #     ("Mia Davis", "Hello Benjamin..."),
    #     ("Benjamin Jackson", "Hello Mia..."),
    #     ...
    # ]

    # 3. 为每个 turn 构造训练样本
    for turn_num, (speaker, utterance) in enumerate(conversation):
        # 4. 构造 prompt（对话上下文）
        prompt = construct_prompt(
            scenario=episode["scenario"],
            agent_name=speaker,
            agent_background=episode["agents_background"][speaker],
            agent_goal=episode["social_goals"][speaker],
            conversation_history=conversation[:turn_num]  # 历史对话
        )

        # 5. 目标输出（GPT-4 的回复）
        target_output = utterance

        # 6. 保存为训练样本
        sft_data.append({
            "input": prompt,
            "output": target_output
        })
```

**输出数据格式：** `sft_data.json`

```json
[
    {
        "input": "Imagine you are Mia Davis...\n[完整的场景、背景、目标、对话历史]\nYou are at Turn #0.\nPlease generate your response...",
        "output": "{\"action_type\": \"speak\", \"argument\": \"Hello Benjamin, great minds think alike...\"}"
    },
    {
        "input": "Imagine you are Mia Davis...\n[包含 Turn #0 的对话历史]\nYou are at Turn #1.\nPlease generate your response...",
        "output": "{\"action_type\": \"speak\", \"argument\": \"I appreciate your understanding...\"}"
    }
]
```

### SFT 训练

**脚本：** `scripts/train_sft.py`

**训练命令：**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file ./accelerate_config_sft.yaml \
  ./train_sft.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --train_batch_size 2 \
    --accumulation_steps 8 \
    --num_epochs 500 \
    --use_lora \  # 使用 LoRA 参数高效微调
    --sft_data_path ../data/sft_data.json \
    --checkpoint_dir ../sft_checkpoints_qwen2.5-7b
```

**训练目标：**

```
Minimize: CrossEntropyLoss(model_output, target_output)
```

模型学习预测 GPT-4 的输出。

**输出：** `sft_checkpoints_qwen2.5-7b/` (SFT 模型检查点)

---

## 阶段 2: Reward Model 训练

### 目标

训练一个 Reward Model，能够自动给每句话打分（不需要每次都调用 GPT-4）。

### 为什么需要 Reward Model？

- ❌ 在 RL 训练中，每生成一句话都调用 GPT-4 评分太慢太贵
- ✅ 训练一个小模型（Reward Model），快速预测 reward
- ✅ Reward Model 学习 GPT-4 的评分标准

### 数据准备流程

这是最关键的部分！分为两步：

#### Step 2.1: LLM Attribution 标注

**脚本：** `scripts/annotate/sample_episodes_and_annotate.py`

**流程：**

```python
# 1. 加载 GPT-4 对话数据
episodes = load_episodes("sotopia_pi_bc_episodes.jsonl")

# 2. 为每个 episode 添加整体得分
for episode in episodes:
    # GPT-4 评估整个对话的表现
    scores = {
        "Mia Davis": 9.0,  # goal 完成度
        "Benjamin Jackson": 7.0
    }
    episode["scores"] = scores

# 3. 使用 GPT-4 进行 attribution 标注
for episode in episodes:
    conversation = parse_conversation(episode)

    for agent in ["Mia Davis", "Benjamin Jackson"]:
        # 构造 Attribution Prompt
        prompt = f"""
        评估 {agent} 的每句话对目标的贡献度...
        [完整的 attribution prompt]
        """

        # 调用 GPT-4
        attribution_scores = call_gpt4(prompt)
        # {
        #     "Utterance 0 by Mia Davis": 3,
        #     "Utterance 1 by Mia Davis": 1,
        #     "Utterance 2 by Mia Davis": 0
        # }

        # 保存 attribution 结果
        save_attributions(episode, agent, attribution_scores)
```

**输出：** `sotopia_pi_bc_episodes_annotated.jsonl`

```json
{
    "episode_id": "01J1234567890ABCDEFG",
    "agent": "Mia Davis",
    "goal": "Negotiate to dress up as the character...",
    "attributed_utterances": {
        "Utterance 0 by Mia Davis": ["Hello Benjamin...", {"reward": 9.0, "attribution": 3}],
        "Utterance 1 by Mia Davis": ["I appreciate...", {"reward": 3.0, "attribution": 1}],
        "Utterance 2 by Mia Davis": ["Absolutely...", {"reward": 0.0, "attribution": 0}]
    },
    "goal_score": 9.0
}
```

#### Step 2.2: 转换为 Reward Model 训练数据

**脚本：** `scripts/data_process/process_annotation_direct_attribution.py`

**关键代码：**

```python
def calc_reward(utter_attrib: float, goal_score: float) -> float:
    """
    计算 reward

    Formula: reward = (attribution / 3) * goal_score
    """
    if utter_attrib == -1:
        reward = -1.0  # 无效标记
    else:
        reward = utter_attrib / 3 * goal_score
    return reward

# 处理每个标注的话语
for d in attributed_data:
    reward_data.append({
        "input": d['prompt'],      # 对话上下文
        "output": d['result'],     # 模型生成的话语
        "value": calc_reward(      # 这句话的 reward
            d['attribution'],      # GPT-4 给的 attribution (0-3)
            d['goal_score']        # 整体目标完成度 (0-10)
        )
    })
```

**输出：** `sotopia_pi_bc_episodes_reward.json`

```json
[
    {
        "input": "Imagine you are Mia Davis...\nYou are at Turn #0...",
        "output": "{\"action_type\": \"speak\", \"argument\": \"Hello Benjamin...\"}",
        "value": 9.0
    },
    {
        "input": "Imagine you are Mia Davis...\nYou are at Turn #1...",
        "output": "{\"action_type\": \"speak\", \"argument\": \"I appreciate...\"}",
        "value": 3.0
    },
    {
        "input": "Imagine you are Mia Davis...\nYou are at Turn #2...",
        "output": "{\"action_type\": \"speak\", \"argument\": \"Absolutely...\"}",
        "value": 0.0
    }
]
```

### Reward Model 训练

**脚本：** `scripts/train_rm.py`

**模型架构：**

```python
from transformers import AutoModelForSequenceClassification

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    num_labels=1  # 输出一个 scalar reward
)
```

**训练目标：**

```
Minimize: MSELoss(predicted_reward, target_reward)

predicted_reward = reward_model(input + output)  # 模型预测
target_reward = value  # GPT-4 标注的 reward
```

**训练命令：**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file ./accelerate_config_rm.yaml \
  ./train_rm.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --learning_rate 1e-5 \
    --max_length 4096 \
    --train_batch_size 1 \
    --num_epochs 30 \
    --reward_data_path ../data/sotopia_pi_bc_episodes_reward.json \
    --checkpoint_dir ../rm_checkpoints_qwen2.5-7b
```

**输出：** `rm_checkpoints_qwen2.5-7b/` (Reward Model 检查点)

**Reward Model 的作用：**

```python
# 训练完成后，可以快速预测任何话语的 reward
context = "对话上下文..."
utterance = "模型生成的话语..."
reward = reward_model(context + utterance)  # 输出: 8.5
```

---

## 阶段 3: GRPO 强化学习训练

### 目标

使用 Reward Model 指导 SFT Model 进行强化学习优化，提升社交对话能力。

### GRPO 算法简介

**GRPO (Group Reward Policy Optimization):**
- 是 PPO 的变体
- 每个 prompt 生成多个 response (如 16 个)
- 使用组内对比的方式计算 advantage
- 更稳定，不需要 value network

### 数据准备

**输入：** 新的对话场景（不需要标注）

**格式：** `grpo_data.json`

```json
[
    {
        "input": "Imagine you are Alice...\n[场景、背景、目标]\nYou are at Turn #0.\nPlease generate your response..."
    },
    {
        "input": "Imagine you are Bob...\n[另一个场景]\nYou are at Turn #0..."
    }
]
```

### GRPO 训练流程

**脚本：** `scripts/train_grpo.py`

**核心流程：**

```python
# 1. 加载模型
policy_model = load_sft_model("../sft_checkpoints_qwen2.5-7b/")
reward_model = load_reward_model("../rm_checkpoints_qwen2.5-7b/")

# 2. GRPO 训练循环
for batch in grpo_dataset:
    prompt = batch["input"]  # 对话上下文

    # 3. 生成多个候选 response
    responses = []
    for _ in range(16):  # 每个 prompt 生成 16 个 response
        response = policy_model.generate(prompt)
        responses.append(response)

    # 4. 使用 Reward Model 评分
    rewards = []
    for response in responses:
        reward = reward_model(prompt + response)
        rewards.append(reward)

    # rewards = [8.5, 6.2, 9.1, 7.3, ..., 5.8]  (16 个 reward)

    # 5. 计算 advantage (组内对比)
    mean_reward = mean(rewards)
    advantages = [r - mean_reward for r in rewards]
    # advantages = [0.3, -2.0, 1.5, -0.9, ..., -2.4]

    # 6. PPO 更新策略
    for response, advantage in zip(responses, advantages):
        # 如果 advantage > 0，增加生成这个 response 的概率
        # 如果 advantage < 0，减少生成这个 response 的概率

        old_logprob = old_policy_model.log_prob(prompt, response)
        new_logprob = policy_model.log_prob(prompt, response)

        ratio = exp(new_logprob - old_logprob)
        clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)

        loss = -min(ratio * advantage, clipped_ratio * advantage)
        loss.backward()

    # 7. 更新参数
    optimizer.step()
```

**训练命令：**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file ./accelerate_config_grpo.yaml \
  ./train_grpo.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --policy_adapter_path ../sft_checkpoints_qwen2.5-7b/best-checkpoints \
    --reward_adapter_path ../rm_checkpoints_qwen2.5-7b/best-checkpoints \
    --learning_rate 5e-6 \
    --num_generations 16 \  # 每个 prompt 生成 16 个候选
    --beta 0.04 \           # KL 散度惩罚系数
    --num_train_epochs 3 \
    --grpo_data_path ../data/grpo_data.json \
    --checkpoint_dir ../grpo_checkpoints_qwen2.5-7b
```

**输出：** `grpo_checkpoints_qwen2.5-7b/` (最终的 RL Agent)

---

## 完整对话示例：Reward 计算全流程

### 场景设定

**场景：** Mia Davis 和 Benjamin Jackson 都想在电影马拉松上扮演同一个角色

**Mia 的目标：** 说服 Benjamin 让她扮演这个角色

**Mia 的背景：** 50岁的高中校长，性格果断直接，最近经历了一些挑战

**Benjamin 的背景：** 24岁的环保活动家，富有表达力和想象力

### 完整对话

```
Turn #0
Mia: "Hello Benjamin, great minds think alike, as we both want to dress up
      as the same character today! However, I've been going through some
      personal challenges recently and I feel that embodying this character
      would provide me with a much-needed boost. Can we make a compromise?"

Turn #0
Benjamin: "Hello Mia, I understand the connection you are seeking with the
           character and I'd never want to ruin that experience for you.
           How about a compromise? We could swap characters for different
           screenings. Does that sound good to you?"

Turn #1
Mia: "I appreciate your understanding, Benjamin. It's comforting to have a
      friend like you. Switching characters for different screenings sounds
      fair, and it would also bring some variety into our tradition. Let's
      do that, I feel good about this compromise."

Turn #1
Benjamin: "I'm glad we reached a compromise, Mia. It's not just about
           dressing up but also about the bond we're creating through these
           shared experiences. Looking forward to the movie marathon with you!"

Turn #2
Mia: "Absolutely, Benjamin. Our shared experiences do make our bond stronger
      and these screenings more meaningful. Shall we get ready then? It's
      going to be a magical movie marathon!"

Turn #2
Benjamin: "Definitely, Mia! Let's weave some magic into this marathon.
           See you at the movies!"

Turn #3
Mia: "I'm excited too, Benjamin! See you at the marathon. Let's make it
      unforgettable!"
```

### 第一步：整体评分

**由 GPT-4 或人类评估整个对话的表现**

**Mia 的维度得分：**

```python
scores = {
    "goal": 9.0,              # 目标完成度：9/10（达成了妥协）
    "relationship": 8.5,      # 关系维护：8.5/10（关系变得更好）
    "believability": 8.0,     # 可信度：8.0/10（表现得很真实）
    "social_rules": 10.0,     # 社交规范：10/10（完全礼貌）
}
```

我们主要关注 **goal = 9.0**

### 第二步：Attribution 标注

**GPT-4 评估 Mia 的每句话对目标的贡献度**

**Attribution Prompt (发送给 GPT-4):**

```
Reward Attribution Instructions for LLMs

Your task is to evaluate the importance of each utterance...

4. Chosen Agent for Evaluation:
Mia Davis

5. Agent's Goal:
Negotiate with your friend to let you dress up as the character this time

7. Conversation History:
Utterance 0 by Mia Davis: "Hello Benjamin, great minds think alike..."
Utterance 0 by Benjamin Jackson: "Hello Mia, I understand..."
Utterance 1 by Mia Davis: "I appreciate your understanding..."
Utterance 1 by Benjamin Jackson: "I'm glad we reached a compromise..."
Utterance 2 by Mia Davis: "Absolutely, Benjamin. Our shared experiences..."
Utterance 2 by Benjamin Jackson: "Definitely, Mia!..."
Utterance 3 by Mia Davis: "I'm excited too, Benjamin!..."

8. Dimension to be Evaluated:
goal

Please format your response as JSON...
```

**GPT-4 的 Attribution 输出：**

```json
{
    "Utterance 0 by Mia Davis": 3,
    "Utterance 1 by Mia Davis": 1,
    "Utterance 2 by Mia Davis": 0,
    "Utterance 3 by Mia Davis": 0
}
```

**推理逻辑：**

| Turn | 话语内容 | Attribution | 推理 |
|------|---------|-------------|------|
| 0 | "Hello Benjamin... Can we make a compromise?" | **3** | ⭐ 最关键的话语！<br>- 直接提出了核心请求<br>- 给出了有说服力的理由（个人挑战）<br>- 主动提出妥协，展现了谈判意愿<br>- 这句话直接推动了 Benjamin 的积极回应 |
| 1 | "I appreciate your understanding..." | **1** | 有一定贡献<br>- 接受了 Benjamin 的妥协方案<br>- 确认了协议<br>- 但目标在 Turn 0 后已基本确定 |
| 2 | "Absolutely, Benjamin..." | **0** | 无实质贡献<br>- 纯粹的礼貌性回应<br>- 对目标完成度无影响<br>- 只是社交性的闲聊 |
| 3 | "I'm excited too, Benjamin!" | **0** | 无实质贡献<br>- 对话已经结束<br>- 纯粹的客套话 |

### 第三步：计算 Reward

**公式：** `reward = (attribution / 3) * goal_score`

**计算每句话的 reward：**

```python
# 参数
goal_score = 9.0  # Mia 的目标完成度
scale = 3         # 使用 3-scale

# Turn #0 的 reward
attribution_0 = 3
reward_0 = (3 / 3) * 9.0 = 1.0 * 9.0 = 9.0

# Turn #1 的 reward
attribution_1 = 1
reward_1 = (1 / 3) * 9.0 = 0.333 * 9.0 = 3.0

# Turn #2 的 reward
attribution_2 = 0
reward_2 = (0 / 3) * 9.0 = 0.0 * 9.0 = 0.0

# Turn #3 的 reward
attribution_3 = 0
reward_3 = (0 / 3) * 9.0 = 0.0 * 9.0 = 0.0
```

### 第四步：构造训练数据

**SFT 训练数据：**

```json
[
    {
        "input": "Imagine you are Mia Davis...\n[场景、背景、目标]\nYou are at Turn #0...",
        "output": "{\"action_type\": \"speak\", \"argument\": \"Hello Benjamin, great minds think alike...\"}"
    },
    {
        "input": "Imagine you are Mia Davis...\n[包含 Turn #0 历史]\nYou are at Turn #1...",
        "output": "{\"action_type\": \"speak\", \"argument\": \"I appreciate your understanding...\"}"
    }
]
```

**Reward Model 训练数据：**

```json
[
    {
        "input": "Imagine you are Mia Davis...\nYou are at Turn #0...",
        "output": "{\"action_type\": \"speak\", \"argument\": \"Hello Benjamin...\"}",
        "value": 9.0
    },
    {
        "input": "Imagine you are Mia Davis...\nYou are at Turn #1...",
        "output": "{\"action_type\": \"speak\", \"argument\": \"I appreciate...\"}",
        "value": 3.0
    },
    {
        "input": "Imagine you are Mia Davis...\nYou are at Turn #2...",
        "output": "{\"action_type\": \"speak\", \"argument\": \"Absolutely...\"}",
        "value": 0.0
    }
]
```

### 第五步：在 GRPO 中使用

**训练时的流程：**

```python
# 1. 给定新场景（类似但不同的对话情境）
prompt = """
Imagine you are Alice...
Scenario: Negotiating with a colleague about project leadership
You are at Turn #0.
Please generate your response...
"""

# 2. 策略模型生成 16 个候选 response
responses = policy_model.generate(prompt, num_samples=16)
# responses = [
#     "{\"action_type\": \"speak\", \"argument\": \"Hi Bob, I think I should lead...\"}",
#     "{\"action_type\": \"speak\", \"argument\": \"Bob, let's discuss leadership...\"}",
#     "{\"action_type\": \"speak\", \"argument\": \"I believe my experience...\"}",
#     ...  (13 more)
# ]

# 3. Reward Model 给每个 response 打分
rewards = []
for response in responses:
    reward = reward_model(prompt + response)
    rewards.append(reward)

# rewards = [8.5, 6.2, 9.1, 7.3, 5.8, 8.9, 7.7, 6.5,
#            9.3, 7.1, 6.8, 8.2, 7.9, 6.4, 8.7, 7.5]

# 4. 计算 advantage (相对于组内平均)
mean_reward = sum(rewards) / 16  # = 7.6
advantages = [r - mean_reward for r in rewards]
# advantages = [0.9, -1.4, 1.5, -0.3, -1.8, 1.3, 0.1, -1.1,
#               1.7, -0.5, -0.8, 0.6, 0.3, -1.2, 1.1, -0.1]

# 5. 根据 advantage 更新策略
# advantage > 0 的 response 会被鼓励（增加概率）
# advantage < 0 的 response 会被抑制（减少概率）

# 例如:
# response[2]: "let's discuss leadership..."
#   → reward=9.1, advantage=1.5 → 增加生成概率！✅
# response[4]: "I think I should lead..."
#   → reward=5.8, advantage=-1.8 → 减少生成概率！❌
```

### 为什么这样设计有效？

**Reward 的含义：**

```
reward = 9.0  → 这句话对目标的贡献非常大！模型应该多生成这样的话
reward = 3.0  → 这句话有一定贡献，但不是最重要的
reward = 0.0  → 这句话没用，浪费 token，应该避免
```

**训练效果：**

经过 GRPO 训练后，模型会学会：
- ✅ 生成高 reward 的话语（直接推动目标的话）
- ✅ 减少低 reward 的话语（客套话、废话）
- ✅ 在合适的时机说合适的话
- ✅ 平衡多个社交维度（目标、关系、礼貌）

---

## 关键代码解析

### 1. Attribution 计算

**文件：** `prompter/direct_attribution_generic_function.py:180-203`

```python
def get_attribution_single_conv(
    conversation,      # [("Mia", "Hello..."), ("Bob", "Hi..."), ...]
    agent,            # "Mia Davis"
    goals,            # {"Mia Davis": "Negotiate...", ...}
    episode,          # 完整 episode 数据
    rewards,          # {"Mia Davis": {"goal": 9.0, ...}}
    llm_name,         # "gpt-4o"
    attribution_instruction_name  # "default-goal"
):
    # 解析配置
    scale, dimension = attribution_instruction_name.split("-")

    # 获取维度得分
    dim_score = rewards[agent][dimension]  # 9.0

    # 构建 Attribution Prompt
    prompt = get_single_attribution_prompt(
        conversation, agent, goals[agent],
        episode["agents_background"][agent],
        dimension, scale
    )

    # 调用 LLM 获取 attribution
    attribution_scores = assign_attributions_for_conversation(
        prompt, conversation, agent, llm_name
    )
    # → {"Utterance 0 by Mia": 3, "Utterance 1 by Mia": 1, ...}

    # 计算 reward
    attribution_rewards = calc_attributed_reward(
        attribution_scores, scale, dim_score
    )
    # → {"Utterance 0 by Mia": {"reward": 9.0, "attribution": 3}, ...}

    return attribution_rewards
```

### 2. Reward 计算

**文件：** `scripts/data_process/process_annotation_direct_attribution.py:63-68`

```python
def calc_reward(utter_attrib: float, goal_score: float) -> float:
    """
    计算单个话语的 reward

    Args:
        utter_attrib: attribution 分数 (0-3)
        goal_score: 目标完成度 (0-10)

    Returns:
        reward: 最终奖励值

    Examples:
        >>> calc_reward(3, 9.0)
        9.0
        >>> calc_reward(1, 9.0)
        3.0
        >>> calc_reward(0, 9.0)
        0.0
    """
    if utter_attrib == -1:
        reward = -1.0  # 无效标记
    else:
        reward = utter_attrib / 3 * goal_score
    return reward
```

### 3. GRPO Dataset

**文件：** `sotopia_rl/data.py`

```python
class GRPODataset(Dataset):
    def __init__(self, data_path, tokenizer, template, max_length):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.data[idx]

        # 渲染 prompt 模板
        prompt = self.template.render(
            scenario=item["scenario"],
            agent_name=item["agent_name"],
            agent_background=item["agent_background"],
            agent_goal=item["agent_goal"],
            conversation_history=item["conversation_history"]
        )

        # Tokenize
        tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }
```

### 4. GRPO Trainer 核心

**文件：** `sotopia_rl/grpo_trainer.py:97-140`

```python
def _setup_policy_models(self):
    # 加载 SFT 模型作为初始策略
    base_policy = AutoModelForCausalLM.from_pretrained(
        self.args.model_name,
        quantization_config=self.quant_config
    )

    # 加载 SFT LoRA 权重
    self.policy = PeftModelForCausalLM.from_pretrained(
        base_policy,
        self.args.policy_adapter_path
    )

def _setup_classification_models(self):
    # 加载 Reward Model
    base_reward = AutoModelForSequenceClassification.from_pretrained(
        self.args.model_name,
        num_labels=1,  # 输出一个 scalar reward
        quantization_config=self.quant_config
    )

    # 加载 Reward Model LoRA 权重
    self.reward_model = PeftModelForSequenceClassification.from_pretrained(
        base_reward,
        self.args.reward_adapter_path
    )

def _setup_grpo_trainer(self):
    config = GRPOConfig(
        num_generations=16,  # 每个 prompt 生成 16 个候选
        beta=0.04,           # KL 散度惩罚
        learning_rate=5e-6,
        num_train_epochs=3,
        per_device_train_batch_size=1,
    )

    self.grpo_trainer = GRPOTrainer(
        model=self.policy,
        reward_model=self.reward_model,
        args=config,
        train_dataset=self.train_dataset,
        tokenizer=self.tokenizer
    )
```

---

## 总结

### 完整流程回顾

```
┌─────────────────────────────────────────────────────────────┐
│ 1. GPT-4 Self-Play                                          │
│    → 收集高质量对话数据                                     │
│    → sotopia_pi_episodes.jsonl                              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 2. SFT 训练                                                 │
│    → 用 GPT-4 数据训练 Qwen2.5-7B                          │
│    → 学会像 GPT-4 一样对话                                 │
│    → sft_checkpoints_qwen2.5-7b/                            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 3. GPT-4 Attribution 标注                                   │
│    → 用 GPT-4 给每句话打 attribution 分数                  │
│    → 计算 reward = (attribution/3) * goal_score            │
│    → sotopia_pi_bc_episodes_reward.json                     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 4. Reward Model 训练                                        │
│    → 用 attribution 数据训练 Reward Model                  │
│    → 学会预测任何话语的 reward                             │
│    → rm_checkpoints_qwen2.5-7b/                             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 5. GRPO 强化学习                                            │
│    → 用 Reward Model 指导策略优化                          │
│    → 每个 prompt 生成 16 个候选                            │
│    → 根据 reward 更新策略                                  │
│    → grpo_checkpoints_qwen2.5-7b/                           │
└─────────────────────────────────────────────────────────────┘
```

### 关键公式

```python
# Reward 计算
reward = (attribution / scale) * dimension_score

# 具体例子
reward = (3 / 3) * 9.0 = 9.0  # 最关键的话语
reward = (1 / 3) * 9.0 = 3.0  # 有一定贡献
reward = (0 / 3) * 9.0 = 0.0  # 无贡献

# GRPO Advantage
advantage = reward - mean(group_rewards)

# PPO Loss
ratio = exp(new_logprob - old_logprob)
clipped_ratio = clip(ratio, 1-0.2, 1+0.2)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

### 核心创新点

1. **Utterance-level Reward**
   - 不是整个对话一个分数
   - 每句话都有独立的 reward
   - 精确的信用分配

2. **Attribution-based**
   - 使用强大的 LLM (GPT-4) 作为评估器
   - 评估每句话对目标的贡献度
   - 无需人工标注

3. **Multi-dimensional**
   - 从多个社交维度评估
   - goal, relationship, believability, social_rules...
   - 可以组合优化

4. **Efficient RL**
   - 训练 Reward Model 替代 GPT-4
   - 快速预测 reward
   - 支持大规模 RL 训练

### 训练数据量级

```
GPT-4 Episodes:     ~10K episodes
SFT Data:           ~50K utterances
RM Data:            ~50K utterances (with rewards)
GRPO Data:          ~20K scenarios
```

### 最终效果

经过完整训练的模型能够：
- ✅ 理解社交场景和目标
- ✅ 生成高 reward 的话语
- ✅ 平衡多个社交维度
- ✅ 在合适的时机说合适的话
- ✅ 达到接近或超过 GPT-4 的社交对话能力

---

## 参考文献

- **论文：** [Sotopia-RL: Reward Design for Social Intelligence](https://arxiv.org/abs/2508.03905)
- **代码：** [GitHub - sotopia-rl](https://github.com/sotopia-lab/sotopia-rl)
- **模型：** [HuggingFace - sotopia-rl-qwen-2.5-7B-grpo](https://huggingface.co/ulab-ai/sotopia-rl-qwen-2.5-7B-grpo)

---

**生成时间：** 2025-11-18
**版本：** 2.0
**作者：** Claude (Anthropic)
