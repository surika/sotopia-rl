# Sotopia-RL Reward 计算机制详解

## 📋 目录

1. [概述](#概述)
2. [核心概念](#核心概念)
3. [Reward 计算公式](#reward-计算公式)
4. [两阶段 LLM 评分系统](#两阶段-llm-评分系统)
5. [Prompt 设计详解](#prompt-设计详解)
6. [代码实现流程](#代码实现流程)
7. [完整示例](#完整示例)
8. [多维度评分系统](#多维度评分系统)
9. [总结](#总结)

---

## 概述

**Sotopia-RL** 使用基于 **LLM 评估的多维度 Attribution (贡献度分配)** 来计算细粒度的 reward，专门用于训练具有社交能力的对话智能体。

### 核心思想

不是给整个对话一个笼统的 reward，而是：
- ✅ **话语级别 (Utterance-level)**：每句话都有自己的 reward
- ✅ **基于归因 (Attribution-based)**：评估每句话对目标的贡献度
- ✅ **多维度 (Multi-dimensional)**：从多个社交维度评估表现

### 技术栈

```python
评估器: GPT-4 / Claude (作为评委给对话打分)
RL 框架: TRL (Transformers Reinforcement Learning)
算法: GRPO (Group Reward Policy Optimization) / PPO
```

---

## 核心概念

### 什么是 Attribution？

**Attribution (贡献度)** 是指每个话语对最终目标达成的贡献程度。

**类比：**
- 假设一场足球比赛赢了 3:0
- 不是所有球员都得到相同的功劳
- 进球的前锋得分高，助攻的中场得分中等，守门员得分低
- Attribution 就是量化每个球员（话语）的贡献

### 为什么需要 Utterance-level Reward？

传统 RL 只给整个对话一个 reward，问题在于：
- ❌ **信用分配困难**：不知道哪句话真正有用
- ❌ **训练信号稀疏**：只有对话结束才有反馈
- ❌ **无法精确优化**：好话和坏话被平均化

Sotopia-RL 的解决方案：
- ✅ **每句话独立评分**：精确的信用分配
- ✅ **密集的训练信号**：每个 turn 都有反馈
- ✅ **可解释性强**：知道为什么某句话重要

---

## Reward 计算公式

### 基础公式

```python
reward = (utterance_attribution / scale) * dimension_score
```

### 三个核心参数详解

#### 1. **utterance_attribution (话语贡献度)**

**定义：** LLM 评估器给每个话语打的"重要性分数"

**取值范围：**
| 标度 | 取值范围 | 分母 (scale) |
|------|----------|--------------|
| 3-scale (default) | 0, 1, 2, 3 | 3 |
| 5-scale | 0, 1, 3, 5 | 5 |
| 10-scale | 0, 1, 5, 10 | 10 |

**打分标准 (3-scale)：**
```
0 分 → 对目标/维度无影响
1-2 分 → 轻微到中等影响 (minor to moderate impact)
2-3 分 → 较大影响 (major impact)
3 分 → 显著影响 (significant impact)
```

**获取方式：** 使用 GPT-4 等强大的 LLM 作为评估器，输入完整对话和目标，输出每句话的 attribution 分数。

#### 2. **scale (标度因子/归一化分母)**

**定义：** 将 attribution 归一化到 0-1 范围的除数

**作用：**
```python
attribution = 3, scale = 3 → 3/3 = 1.0 (100% 贡献)
attribution = 1, scale = 3 → 1/3 = 0.33 (33% 贡献)
attribution = 0, scale = 3 → 0/3 = 0.0 (无贡献)
```

**代码映射：**
```python
denominator = {
    "default": 3,      # 3分制
    "5_scale": 5,      # 5分制
    "10_scale": 10     # 10分制
}
```

#### 3. **dimension_score (维度得分)**

**定义：** 整个对话在某个维度上的最终表现得分

**取值范围：** 通常是 0-10 分

**示例：**
```python
episode["scores"]["Mia Davis"] = 9.0  # Mia 的目标完成度
rewards["Mia Davis"] = {
    "goal": 9.0,                    # 目标完成度
    "relationship": 7.5,            # 关系维护
    "believability": 8.0,           # 可信度
    "social_rules": 10.0            # 社交规范
}
```

### 计算示例

**场景：** Mia 想说服 Benjamin 让她穿某个角色的衣服

```python
# 阶段1：整体评分
goal_score = 9.0  # Mia 的目标完成度为 9/10

# 阶段2：LLM 评估每句话的 attribution
attributions = {
    "Utterance 0 by Mia": 3,  # 提出请求和理由 - 最关键
    "Utterance 1 by Mia": 1,  # 接受妥协 - 有一定贡献
    "Utterance 2 by Mia": 0   # 礼貌性结束 - 无实质贡献
}

# 阶段3：计算每句话的 reward (使用 3-scale)
rewards = {
    "Utterance 0": (3/3) * 9.0 = 9.0,   ✅ 最高奖励
    "Utterance 1": (1/3) * 9.0 = 3.0,   ✅ 中等奖励
    "Utterance 2": (0/3) * 9.0 = 0.0    ❌ 无奖励
}
```

---

## 两阶段 LLM 评分系统

Sotopia-RL 使用 **两次独立的 LLM 调用** 来完成完整的 reward 计算：

```
┌─────────────────────────────────────────┐
│  阶段 1: 整体维度评分 (Dimension Score) │
│  Input: 完整对话 + Agent 目标          │
│  Output: dimension_score (0-10)        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  阶段 2: 话语贡献度评分 (Attribution)   │
│  Input: 完整对话 + 目标 + 维度描述     │
│  Output: attribution_scores (0-3/5/10) │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  计算 Reward                            │
│  reward = (attribution/scale) * score   │
└─────────────────────────────────────────┘
```

### 阶段 1：整体维度评分

**目的：** 评估整个对话在某个维度上的整体表现

**评估维度：**

| 维度 | 英文名 | 描述 | 评分范围 |
|------|--------|------|----------|
| 目标完成度 | goal | 主要社交目标的达成程度 | 0-10 |
| 可信度 | believability | Agent 表现得多像真人 | 0-10 |
| 关系维护 | relationship | 社交关系的改善/恶化程度 | 0-10 |
| 知识获取 | knowledge | 获得新信息的程度 | 0-10 |
| 秘密保护 | secret | 秘密保护得有多好 | 0-10 |
| 社交规范 | social_rules | 遵守道德和法律的程度 | 0-10 |
| 经济收益 | financial_benefits | 财务/物质上的收益 | 0-10 |
| 对话行为 | conversation_behavior | 避免重复、合理结束 | 0-10 |

**Prompt 结构（简化版）：**

```
评估以下对话中 {agent} 在 {dimension} 维度上的表现：

Agent 的目标: {goal}
Agent 的背景: {background}

对话内容:
Turn 0 - Alice: "..."
Turn 1 - Bob: "..."
...

请根据以下标准给出 0-10 的评分：
- 0-3分: 表现差/未达成
- 4-6分: 中等表现
- 7-9分: 良好表现
- 10分: 完美表现

输出格式: {"score": 9.0}
```

**输出示例：**
```json
{
    "goal": 9.0,
    "relationship": 7.5,
    "believability": 8.0
}
```

### 阶段 2：话语贡献度评分

**目的：** 评估每一句话对最终维度得分的贡献程度

**核心文件：** `prompter/generic_templates.py:104-141`

---

## Prompt 设计详解

### 完整的 Attribution Prompt 模板

```python
DIRECT_ATTRIBUTION_TEMPLATE = """
Reward Attribution Instructions for LLMs

Your task is to evaluate the importance of each utterance in a conversation
between two agents on a certain dimension of evaluation.

1. Input Context:
   - You will receive the dialogue history between two conversational agents
     each with their own social goal.
   - You will be provided with the social goal of one of the agents.
   - You will be provided with the dimension to be evaluated and the
     description of the dimension.

2. Objective:
   - Assign an importance value to each utterance (identified by the agent's
     name and utterance number) based on its contribution to the achievement
     on the provided dimension.
   - Consider both the individual utterance and the responses from the other
     agent, as both affect the final outcome.

3. General Scoring Scale:
{scoring_guidelines}

4. Chosen Agent for Evaluation:
{agent}

5. Agent's Goal:
{goal}

6. Agent's Background:
{agent_background}

7. Conversation History:
{conversation}

8. Dimension to be Evaluated:
{dimension}

9. Dimension Description:
{dimension_description}

10. Formatting Instructions:
{formatting_instructions}
"""
```

### 实际填充后的完整示例

```
Reward Attribution Instructions for LLMs

Your task is to evaluate the importance of each utterance in a conversation
between two agents on a certain dimension of evaluation.

1. Input Context:
   - You will receive the dialogue history between two conversational agents.
   - You will be provided with the social goal of one of the agents.
   - You will be provided with the dimension to be evaluated.

2. Objective:
   - Assign an importance value to each utterance based on its contribution
     to the achievement on the provided dimension.
   - Consider both the individual utterance and the responses from the other
     agent.

3. General Scoring Scale:
   - If an utterance has no impact on the final achievement on the dimension,
     assign it an importance of 0.
   - If an utterance has a minor impact, assign it an importance of 1 to 2.
   - If an utterance has a major impact, assign it an importance of 2 to 3.
   - If an utterance has a significant impact, assign it an importance of 3.

   Note:
   - Please provide a score for each utterance of the chosen agent.
   - Please only assign a score between 0 and 3.

4. Chosen Agent for Evaluation:
Mia Davis

5. Agent's Goal:
Negotiate with your friend to let you dress up as the character this time
(Extra information: you've recently experienced some personal challenges and
you feel that taking on the role of this character would be particularly
meaningful for you)

6. Agent's Background:
Mia Davis is a 50-year-old female high school principal. She/her pronouns.
Personality and values description: Mia Davis, an extraverted stickler for
routines, values tradition and authority. Her decision-making style is
decisive and direct.

7. Conversation History:
Utterance 0 by Mia Davis: Hello Benjamin, great minds think alike, as we
both want to dress up as the same character today! However, I've been going
through some personal challenges recently and I feel that embodying this
character would provide me with a much-needed boost. Can we make a compromise?

Utterance 0 by Benjamin Jackson: Hello Mia, I understand the connection you
are seeking with the character and I'd never want to ruin that experience for
you. How about a compromise? We could swap characters for different screenings.
Does that sound good to you?

Utterance 1 by Mia Davis: I appreciate your understanding, Benjamin. Switching
characters for different screenings sounds fair, and it would also bring some
variety into our tradition. Let's do that.

Utterance 1 by Benjamin Jackson: I'm glad we reached a compromise, Mia.
Looking forward to the movie marathon with you!

Utterance 2 by Mia Davis: Absolutely, Benjamin. Shall we get ready then?
It's going to be a magical movie marathon!

8. Dimension to be Evaluated:
goal

9. Dimension Description:
Goal refers to the reiteration of the agent's social goals and the analysis
of their achievement. A higher score indicates significant progress or
achievement of the stated goals.

DOMAIN SPECIFIC SCORING GUIDELINES:
   - The highest score should be assigned to the utterance that is most
     relevant to the goal.
   - A lower score should be assigned to utterances that are not relevant.
   - If a goal is already achieved, the utterance should not be assigned a
     score higher than 1.

10. Formatting Instructions:
Please format your response as JSON with the following structure:
{
    "Utterance 0 by Mia Davis": 0,
    "Utterance 1 by Mia Davis": 2,
    ...
}

You are supposed to be returning 3 attributions.
```

### LLM 返回的 JSON 结果

```json
{
    "Utterance 0 by Mia Davis": 3,
    "Utterance 1 by Mia Davis": 1,
    "Utterance 2 by Mia Davis": 0
}
```

**推理逻辑：**
- **Utterance 0 = 3**: 直接提出请求和理由，推动了整个谈判 → 显著影响
- **Utterance 1 = 1**: 接受妥协方案，确认了协议但目标已基本达成 → 轻微影响
- **Utterance 2 = 0**: 礼貌性结束对话，对目标完成无实质影响 → 无影响

---

## 代码实现流程

### 文件结构

```
sotopia-rl/
├── prompter/
│   ├── generic_templates.py              # Prompt 模板和维度定义
│   ├── direct_attribution_generic_function.py  # 主要实现
│   ├── attribution_prompting.py          # 批处理和并发
│   └── attribution_methods.py            # 8种 attribution 方法
├── scripts/
│   └── annotate/
│       └── sample_episodes_and_annotate.py  # 生成 LLM 标注
└── data/
    └── sotopia_pi_gpt4_rm_overfit.json   # 训练数据
```

### 核心代码实现

**文件：** `prompter/direct_attribution_generic_function.py:180-203`

```python
def get_attribution_single_conv(
    conversation,      # [("Mia", "Hello..."), ("Bob", "Hi..."), ...]
    agent,            # "Mia Davis"
    goals,            # {"Mia Davis": "Negotiate...", "Bob": "..."}
    episode,          # 完整 episode 数据
    rewards,          # {"Mia Davis": {"goal": 9.0, "relationship": 7.0}}
    llm_name,         # "gpt-4o"
    attribution_instruction_name  # "default-goal"
):
    """
    为单个对话生成 attribution 和 reward
    """

    # ========== Step 1: 解析配置 ==========
    scale, dimension = attribution_instruction_name.split("-")
    # scale="default" (3分制), dimension="goal"

    assert scale in SCALE_GUIDELINE_DICT
    assert dimension in DIMENSION_DESCRIPTION_DICT

    # ========== Step 2: 获取维度得分 ==========
    agent_goal = goals[agent]
    agent_background = episode["agents_background"][agent]

    if dimension == "conversation_behavior":
        dim_score = 10  # 固定为 10
    elif dimension == "goal_barebone":
        dim_score = rewards[agent]["goal"]
    else:
        dim_score = rewards[agent][dimension]
    # dim_score = 9.0

    # ========== Step 3: 构建 Prompt ==========
    prompt = get_single_attribution_prompt(
        conversation=conversation,
        agent=agent,
        agent_goal=agent_goal,
        agent_background=agent_background,
        dimension=dimension,
        scale=scale
    )
    # 返回完整的 Attribution Prompt

    # ========== Step 4: 调用 LLM 获取 Attribution ==========
    attribution_scores = assign_attributions_for_conversation(
        prompt,
        conversation,
        agent,
        llm_name=llm_name
    )
    # attribution_scores = {
    #     "Utterance 0 by Mia Davis": 3,
    #     "Utterance 1 by Mia Davis": 1,
    #     "Utterance 2 by Mia Davis": 0
    # }

    # ========== Step 5: 计算 Reward ==========
    attribution_rewards = calc_attributed_reward(
        attribution_scores,
        scale=scale,      # "default" → denominator=3
        dim_score=dim_score  # 9.0
    )

    # ========== Step 6: 添加元数据 ==========
    for key in attribution_rewards:
        attribution_rewards[key]["dimension"] = dimension
        attribution_rewards[key]["scale"] = scale
        attribution_rewards[key]["dim_score"] = dim_score

    # attribution_rewards = {
    #     "Utterance 0 by Mia Davis": {
    #         "reward": 9.0,
    #         "attribution": 3,
    #         "dimension": "goal",
    #         "scale": "default",
    #         "dim_score": 9.0
    #     },
    #     "Utterance 1 by Mia Davis": {
    #         "reward": 3.0,
    #         "attribution": 1,
    #         "dimension": "goal",
    #         "scale": "default",
    #         "dim_score": 9.0
    #     }
    # }

    return attribution_rewards
```

### Reward 计算函数

**文件：** `prompter/direct_attribution_generic_function.py:151-163`

```python
def calc_reward(utter_attrib: float, scale: str, dim_score: float) -> float:
    """
    计算单个话语的 reward

    Args:
        utter_attrib: attribution 分数 (如 3)
        scale: 标度类型 ("default", "5_scale", "10_scale")
        dim_score: 维度得分 (如 9.0)

    Returns:
        reward: 最终奖励值
    """
    denominator = {"default": 3, "5_scale": 5, "10_scale": 10}[scale]

    if utter_attrib == -1:
        reward = -1.0  # 特殊标记：无效的 attribution
    else:
        reward = utter_attrib / denominator * dim_score

    return reward

def calc_attributed_reward(
    attributed_data: Dict[str, float | int],
    scale: str,
    dim_score: float | int
) -> Dict[str, Dict[str, Any]]:
    """
    为所有话语计算 reward
    """
    utterance_reward_map = {}

    for utterance_key, attribution_value in attributed_data.items():
        utterance_reward_map[utterance_key] = {
            "reward": calc_reward(attribution_value, scale, dim_score),
            "attribution": attribution_value
        }

    return utterance_reward_map
```

### LLM API 调用

**文件：** `prompter/direct_attribution_generic_function.py:112-149`

```python
def assign_attributions_for_conversation(
    prompt: str,
    conversation: list,
    agent: str,
    llm_name: str = "gpt-3.5-turbo"
) -> Dict[str, int]:
    """
    调用 LLM API 获取 attribution 分数，包含重试机制
    """

    # 最多重试 5 次
    for attempt in range(5):
        # 计算该 agent 有多少句话
        uttr_count = sum(1 for speaker, _ in conversation if speaker == agent)

        # 调用 OpenAI/Anthropic/Together API
        response = openai_call(
            prompt + f"\nYou are supposed to be returning {uttr_count} attributions.",
            model=llm_name
        )

        if response is None:
            print("Failed to get response from LLM")
            return {}

        # 解析 JSON 响应
        try:
            result = json.loads(response)

            # 确保所有值都是整数
            for key in result:
                result[key] = int(result[key])

        except (json.JSONDecodeError, ValueError):
            print(f"JSON parsing failed, retrying... (attempt {attempt+1}/5)")
            continue

        # 验证返回的 attribution 数量是否正确
        if len(result) == uttr_count:
            return result  # 成功
        else:
            print(f"Count mismatch: expected {uttr_count}, got {len(result)}")

    # 5 次重试都失败
    return {}
```

---

## 完整示例

### 场景描述

**背景：** Mia Davis 和 Benjamin Jackson 都想在电影马拉松上扮演同一个角色

**Mia 的目标：** 说服 Benjamin 让她扮演这个角色（因为她最近经历了一些挑战）

**对话内容：**

```
Turn 0 - Mia Davis:
"Hello Benjamin, great minds think alike! However, I've been going through
some personal challenges recently and I feel that embodying this character
would provide me with a much-needed boost. Can we make a compromise?"

Turn 0 - Benjamin Jackson:
"Hello Mia, I understand. How about a compromise? We could swap characters
for different screenings. Does that sound good to you?"

Turn 1 - Mia Davis:
"I appreciate your understanding, Benjamin. Switching characters for
different screenings sounds fair. Let's do that."

Turn 1 - Benjamin Jackson:
"I'm glad we reached a compromise, Mia. Looking forward to it!"

Turn 2 - Mia Davis:
"Absolutely, Benjamin. Shall we get ready then?"
```

### Step-by-Step 计算过程

#### Step 1: 整体维度评分

**输入：** 完整对话 + Mia 的目标和背景

**LLM 评估：**
```
Mia 的目标完成度评估：
- 她的目标是让 Benjamin 同意让她穿那个角色
- 最终结果：达成了轮流穿的妥协
- 部分达成了目标，虽然不是完全独占
- 评分：9.0 / 10
```

**输出：**
```python
episode["scores"]["Mia Davis"] = 9.0
```

#### Step 2: 话语 Attribution 评分

**输入：** 完整对话 + Mia 的目标 + Attribution Prompt

**LLM 评估过程：**

**Utterance 0 by Mia:**
```
"...I've been going through some personal challenges...Can we make a compromise?"

分析：
- 直接提出了核心请求
- 给出了有说服力的理由（个人挑战）
- 主动提出妥协，展现了谈判意愿
- 这是整个对话中最关键的话语，直接推动了谈判

评分: 3 / 3 (显著影响)
```

**Utterance 1 by Mia:**
```
"I appreciate your understanding...Switching characters...sounds fair."

分析：
- 接受了 Benjamin 的妥协方案
- 确认了协议，有助于目标达成
- 但目标在 Utterance 0 后已基本确定
- 这句话主要是确认而非推动

评分: 1 / 3 (轻微影响)
```

**Utterance 2 by Mia:**
```
"Absolutely, Benjamin. Shall we get ready then?"

分析：
- 纯粹的礼貌性结束语
- 对目标完成度没有任何实质影响
- 目标已经完全达成，这句话只是闲聊

评分: 0 / 3 (无影响)
```

**输出：**
```json
{
    "Utterance 0 by Mia Davis": 3,
    "Utterance 1 by Mia Davis": 1,
    "Utterance 2 by Mia Davis": 0
}
```

#### Step 3: 计算最终 Reward

**公式：** `reward = (attribution / scale) * dim_score`

**参数：**
- scale = 3 (使用 3-scale)
- dim_score = 9.0 (Mia 的目标完成度)

**计算：**

```python
Utterance 0 reward = (3 / 3) * 9.0 = 1.0 * 9.0 = 9.0
Utterance 1 reward = (1 / 3) * 9.0 = 0.33 * 9.0 = 3.0
Utterance 2 reward = (0 / 3) * 9.0 = 0.0 * 9.0 = 0.0
```

**最终输出：**
```json
{
    "Utterance 0 by Mia Davis": {
        "reward": 9.0,
        "attribution": 3,
        "dimension": "goal",
        "scale": "default",
        "dim_score": 9.0
    },
    "Utterance 1 by Mia Davis": {
        "reward": 3.0,
        "attribution": 1,
        "dimension": "goal",
        "scale": "default",
        "dim_score": 9.0
    },
    "Utterance 2 by Mia Davis": {
        "reward": 0.0,
        "attribution": 0,
        "dimension": "goal",
        "scale": "default",
        "dim_score": 9.0
    }
}
```

### 训练数据格式

这些 reward 会被转换成 Reward Model 的训练数据：

**文件：** `data/sotopia_pi_gpt4_rm_overfit.json`

```json
[
    {
        "input": "Imagine you are Mia Davis...\n[完整的 context 和 prompt]",
        "output": "{\"action_type\": \"speak\", \"argument\": \"Hello Benjamin...\"}",
        "value": 9.0
    },
    {
        "input": "...[Turn 1 的 context]",
        "output": "{\"action_type\": \"speak\", \"argument\": \"I appreciate...\"}",
        "value": 3.0
    },
    {
        "input": "...[Turn 2 的 context]",
        "output": "{\"action_type\": \"speak\", \"argument\": \"Absolutely...\"}",
        "value": 0.0
    }
]
```

---

## 多维度评分系统

### 支持的 9 个维度

| 维度 | 英文 | 描述 | 评分范围 | 文件位置 |
|------|------|------|----------|----------|
| 可信度 | believability | Agent 表现得多像真人 | 0-10 | `generic_templates.py:35-37` |
| 关系维护 | relationship | 社交关系的改善/恶化 | 0-10 | `generic_templates.py:39-41` |
| 知识获取 | knowledge | 获得新信息的程度 | 0-10 | `generic_templates.py:43-45` |
| 秘密保护 | secret | 秘密保护程度 | 0-10 | `generic_templates.py:47-49` |
| 社交规范 | social_rules | 遵守道德和法律 | 0-10 | `generic_templates.py:51-53` |
| 经济收益 | financial_benefits | 财务/物质收益 | 0-10 | `generic_templates.py:55-57` |
| 目标完成 | goal | 主要目标达成度 ⭐ | 0-10 | `generic_templates.py:63-72` |
| 对话行为 | conversation_behavior | 避免重复、合理结束 | 0-10 | `generic_templates.py:74-84` |
| 基本目标 | goal_barebone | 不含其他维度的纯目标 | 0-10 | `generic_templates.py:59-61` |

### 多维度计算示例

**场景：** 商业谈判

```python
# Step 1: 各维度的整体得分
dimension_scores = {
    "goal": 8.0,              # 谈判目标达成度
    "relationship": 7.0,      # 维持了良好关系
    "believability": 9.0,     # 表现得很专业
    "social_rules": 10.0,     # 完全遵守商业礼仪
    "financial_benefits": 6.0 # 获得了一定收益
}

# Step 2: 某个关键话语的 attribution (3-scale)
utterance_5_attributions = {
    "goal": 3,              # 对目标贡献最大
    "relationship": 1,      # 对关系有轻微影响
    "believability": 2,     # 展现了专业性
    "social_rules": 0,      # 对规范无特殊贡献
    "financial_benefits": 2 # 对收益有较大贡献
}

# Step 3: 计算各维度的 reward
rewards = {
    "goal": (3/3) * 8.0 = 8.0,
    "relationship": (1/3) * 7.0 = 2.33,
    "believability": (2/3) * 9.0 = 6.0,
    "social_rules": (0/3) * 10.0 = 0.0,
    "financial_benefits": (2/3) * 6.0 = 4.0
}

# Step 4: 组合策略（可选）
# 方法1: 只使用 goal 维度
final_reward = 8.0

# 方法2: 加权平均
final_reward = 0.5 * 8.0 + 0.2 * 2.33 + 0.15 * 6.0 + 0.15 * 4.0
             = 6.37

# 方法3: 分别训练多个 Reward Model
# 每个维度训练一个独立的 RM，然后在 RL 中组合使用
```

### 8 种不同的 Attribution 方法

**文件：** `prompter/attribution_methods.py`

| 方法 | 描述 | 特点 |
|------|------|------|
| Direct Attribution | LLM 直接评估贡献度 | 最常用，平衡性好 |
| Direct Attribution Normalized | 归一化版本，总和=goal_score | 保证 reward 总量恒定 |
| Direct Attribution Generic | 支持多维度评估 | 灵活性最高 ⭐ |
| Key Utterance | 只标记最关键的话语 | 简化版，binary 标记 |
| Goal Progress | 基于目标进度增量 | 关注每步的进展 |
| Only Response | 仅基于对方的响应质量 | 强调交互效果 |
| Utterance Quality | 评估话语本身的质量 | 侧重语言表达 |
| Discounting / All-the-Same | 时间折扣或平均分配 | Baseline 方法 |

---

## 总结

### 完整的 Reward 计算流程

```
┌─────────────────────────────────────────────────────┐
│  1. 对话完成                                         │
│     → 完整的对话历史 + Agent 目标和背景             │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  2. 阶段1：整体维度评分                             │
│     → 使用 LLM 评估整个对话的表现                   │
│     → 得到 dimension_score (如 goal=9.0)            │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  3. 阶段2：话语贡献度评分                           │
│     → 构建详细的 Attribution Prompt                 │
│     → 发送给 GPT-4/Claude 等 LLM                    │
│     → LLM 返回每句话的 attribution (0-3/5/10)       │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  4. 计算 Reward                                     │
│     → reward = (attribution / scale) * dim_score    │
│     → 示例: (3/3) * 9.0 = 9.0                       │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  5. 保存为训练数据                                   │
│     → {"input": prompt, "output": response,         │
│        "value": reward}                             │
│     → 用于训练 Reward Model                         │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  6. 在 GRPO/PPO 中使用                              │
│     → Reward Model 给新生成的话语打分               │
│     → RL 算法根据 reward 优化策略                   │
│     → 训练更智能的社交对话 agent                    │
└─────────────────────────────────────────────────────┘
```

### 核心优势

1. **细粒度 (Fine-grained)**
   - 每句话都有独立的 reward
   - 精确的信用分配
   - 密集的训练信号

2. **可解释性 (Interpretable)**
   - 知道每句话为什么重要
   - Attribution 分数有明确含义
   - 易于调试和分析

3. **多维度 (Multi-dimensional)**
   - 支持 9 个不同的社交维度
   - 可以同时优化多个目标
   - 灵活的组合策略

4. **基于 LLM (LLM-based)**
   - 利用强大 LLM 的判断能力
   - 无需人工标注
   - 可扩展到大规模数据

### 关键文件索引

```
核心实现：
├── prompter/generic_templates.py                # Prompt 模板和维度定义
├── prompter/direct_attribution_generic_function.py  # 主要实现逻辑
├── prompter/attribution_prompting.py            # 批处理和并发
└── prompter/attribution_methods.py              # 8种 attribution 方法

训练流程：
├── scripts/annotate/sample_episodes_and_annotate.py  # 生成 LLM 标注
├── sotopia_rl/rm_trainer.py                     # Reward Model 训练
├── sotopia_rl/grpo_trainer.py                   # GRPO RL 训练
└── sotopia_rl/ppo_trainer.py                    # PPO RL 训练

数据格式：
└── data/sotopia_pi_gpt4_rm_overfit.json         # 训练数据示例
```

### 公式速查

```python
# 基础公式
reward = (utterance_attribution / scale) * dimension_score

# 参数范围
utterance_attribution: 0-3 (default) / 0-5 / 0-10
scale: 3 / 5 / 10
dimension_score: 0-10

# 示例
reward = (3/3) * 9.0 = 9.0  # 最高奖励
reward = (1/3) * 9.0 = 3.0  # 中等奖励
reward = (0/3) * 9.0 = 0.0  # 无奖励
```

---

## 参考资料

- **论文：** [Sotopia-RL: Reward Design for Social Intelligence](https://arxiv.org/abs/2508.03905)
- **代码仓库：** [GitHub - sotopia-rl](https://github.com/sotopia-lab/sotopia-rl)
- **模型：** [HuggingFace - sotopia-rl-qwen-2.5-7B-grpo](https://huggingface.co/ulab-ai/sotopia-rl-qwen-2.5-7B-grpo)
- **项目主页：** [rl.sotopia.world](https://rl.sotopia.world/)

---

**生成时间：** 2025-11-18
**版本：** 1.0
**作者：** Claude (Anthropic)
