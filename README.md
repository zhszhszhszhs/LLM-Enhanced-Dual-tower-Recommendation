# 大模型增强用户意图推荐

这个 README 以**使用说明**为主，按“从数据到训练”给出最短可执行流程。

## 1. 项目结构（使用视角）

- `script/`：数据处理、调用 LLM、生成标签和 embedding 的脚本。
- `generation/`：Prompt 模板（已按 `beauty / toys / sports` 拆分）。
- `model/`：训练与评估代码（`train.py` + `config.yml`）。
- `dataset/`：数据与中间产物目录（已在 `.gitignore` 中忽略）。

## 2. 环境准备

### 2.1 安装依赖

```bash
pip install torch numpy scipy tqdm pyyaml dashscope
```

### 2.2 配置千问密钥

项目已移除明文密钥，运行前请设置环境变量：

```bash
export DASHSCOPE_API_KEY="your_api_key"
```

## 3. 数据集切换方式（重要）

多个脚本底部都有：

```python
DATASET = 'sports'
```

可选值：

- `beauty`
- `toys`
- `sports`

你只需要改这个变量，脚本会自动切换对应的：

- 输入数据路径 `dataset/{DATASET}/...`
- prompt 路径 `generation/..._{DATASET}.txt`
- 输出路径 `dataset/{DATASET}/processed/...`

## 4. 一次完整流程（推荐顺序）

以下是从原始数据到模型训练的标准流程：

### Step 1：数据预处理与划分

```bash
python script/data_process.py
python script/split_dataset.py
```

### Step 2：生成 LLM 输入

```bash
python script/generate_item_llm_input.py
python script/generate_user_llm_input.py
```

### Step 3：生成画像与标签（需要 API）

```bash
python script/generate_item_profiles.py
python script/generate_user_profiles.py
python script/generate_item_tags.py
```

### Step 4：生成 embedding（需要 API）

```bash
python script/generate_item_embeddings.py
python script/generate_user_embeddings.py
python script/generate_tag_embeddings.py
```

### Step 5：构建聚类/超图相关文件

```bash
python script/cluster_tags_and_analyze.py
python script/generate_item_cluster_matrix.py
```

### Step 6：训练与评估

```bash
python model/train.py --config model/config.yml
```

## 5. 训练前必须检查

训练前请确认以下文件已生成（以 `dataset/<name>/processed/` 为例）：

- `llm_input.json`
- `user_llm_input.json`
- `item_profiles.json`
- `user_profiles.json`
- `item_tags.json`
- `item_embeddings.npy`
- `user_embeddings.npy`
- `tag_embeddings.npy`
- 超图相关矩阵文件（训练配置对应路径可读）

## 6. 常用运行模式说明

很多生成脚本支持两种模式：

- `MODE = 'test'`：随机取一个样本调试，先看输出是否合理。
- `MODE = 'batch'`：全量处理。

断点续跑相关参数：

- `START_INDEX`：从第几个样本继续。
- `APPEND_MODE = True`：追加写入，适合中断后继续。

建议首次先用 `test`，确认格式后再切 `batch`。

## 7. 输出位置

- 日志：`log/<model_name>/`
- 模型权重：`checkpoint/<model_name>/`
- 各类中间结果：`dataset/<name>/processed/`

## 8. 常见问题

- **报错：API 调用失败**
  - 检查 `DASHSCOPE_API_KEY` 是否生效：`echo $DASHSCOPE_API_KEY`
  - 检查配额和网络。
- **报错：文件不存在**
  - 通常是上游脚本未执行，按第 4 节顺序补跑。
- **报错：维度不一致**
  - 检查 embedding 模型版本和脚本里的 `EMBEDDING_DIM` 是否一致（例如 v4 常用 1024）。
- **显存/内存不足**
  - 降低 `model/config.yml` 中 `train.batch_size`、`test.batch_size`。

## 9. 最短上手（sports）

```bash
export DASHSCOPE_API_KEY="your_api_key"
python script/generate_item_llm_input.py
python script/generate_user_llm_input.py
python script/generate_item_profiles.py
python script/generate_user_profiles.py
python script/generate_item_tags.py
python script/generate_item_embeddings.py
python script/generate_user_embeddings.py
python script/generate_tag_embeddings.py
python script/cluster_tags_and_analyze.py
python script/generate_item_cluster_matrix.py
python model/train.py --config model/config.yml
```
