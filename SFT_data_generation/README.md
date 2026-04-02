# teacher-data-gen

这是一个面向 GSM8K SFT 蒸馏任务的 teacher model 数据生成模块。

## 功能说明

- 从 `openai/gsm8k` 的 `main/train` 中读取前3000条样本
- 只使用 `question` 字段
- 调用阿里云百炼的 `qwen3.5-397b-a17b`
- 输出 `success.jsonl`、`bad.jsonl`、`run_stats.json` 和 `run.log`

## 安装

```bash
pip install -e .
```

## API Key 配置

可以直接在config.yaml里填写，或使用环境变量

```bash
set DASHSCOPE_API_KEY=YOUR_KEY_HERE
```

对应配置写法：

```yaml
teacher:
  api_key_env: DASHSCOPE_API_KEY
```

系统 prompt 单独存放在 `prompts/teacher_system_prompt.txt`，可以通过 `teacher.system_prompt_path` 修改路径。

## 运行方式

```bash
python -m teacher_data_gen.main --config configs/teacher_gsm8k.yaml
```

## 输出格式

成功样本会写入 `success.jsonl`，字段包括：

- `id` //数据集本身无id，手动编号
- `question` //题干
- `response.reasoning` //思考过程
- `response.answer` //答案
- `response.text` //思考过程+标签化答案
- `created_at`
- `teacher_model`
- `usage`

其中最终答案始终附在 `response.text` 的末尾：

```text
<final_answer>...</final_answer>
```

过滤样本或失败样本会写入 `bad.jsonl`。

## 说明

- 默认不使用qwen3.5-397b-a17b自带的thinking mode，而是直接要求模型直接输出简洁的 `reasoning` 和 `answer` JSON，这样生成的数据更短、更干净，也更容易解析。不然实测会出现一直思考停不下来的情况，同时原始的thinking里会复述system prompt的一些内容
