# Hungarian Loss - No Object Matching

## 功能说明

**No Object Matching** 是Hungarian Loss的扩展功能，用于处理预测数量超过目标数量的情况。

## 问题背景

在标准Hungarian matching中：
```
预测: [A, B, C, D, E]  (L=5)
目标: [X, Y, Z]        (T=3)

匹配结果: (0→X, 2→Y, 4→Z)  ← 只匹配3对
未匹配: [B, D]  ← 这些预测不参与loss计算！
```

**后果**：
- 未匹配的预测不被惩罚
- 模型倾向于过度预测（insertion rate增加）
- 在之前的实验中：insertion rate从8.53%上升到10.20% (+19.58%)

## 解决方案

启用 `--use_no_object` 后：
```
预测: [A, B, C, D, E]  (L=5)
目标: [X, Y, Z]        (T=3)

匹配结果:
  正常匹配: (0→X, 2→Y, 4→Z)
  No object: (1→∅, 3→∅)  ← 未匹配的预测被标记为"no object"

Loss计算:
  loss_normal = CE(pred[0], X) + CE(pred[2], Y) + CE(pred[4], Z)
  loss_no_obj = CE(pred[1], PAD) + CE(pred[3], PAD)  ← 惩罚它们预测PAD
  total = loss_normal + 0.1 * loss_no_obj
```

**效果**：
- ✅ 减少insertion rate（模型学会何时"停止"预测）
- ✅ 更精确的序列长度控制
- ✅ 更接近DETR的完整实现

## 使用方法

### 参数说明

```bash
--use_no_object          # 启用no object matching
--no_object_weight 0.1   # no object loss的权重（默认0.1）
--allow_null_match       # 在匹配矩阵加入一列 dummy，允许预测直接匹配“空目标”
--no_object_cost 2.0     # dummy 列的匹配成本（默认2.0，成本越低越容易匹配为空）
```

### 训练脚本示例

```bash
# 启用 no object（保持标签原顺序）
./script/train_cslr_hungarian_1.0_shuffled_no_object.sh

# 启用 no object + 空集匹配（允许部分预测显式对齐到PAD）
./script/train_cslr_hungarian_1.0_shuffled_no_object.sh --allow_null_match --no_object_cost 2.0
```

### 手动使用

```bash
deepspeed fine_tuning.py \
  --task CSLR \
  --dataset CSL_Daily \
  --use_hungarian \
  --hungarian_weight 1.0 \
  --use_no_object \           # 启用no object matching
  --no_object_weight 0.1      # no object loss权重
  --allow_null_match \        # 可选：允许部分预测直接匹配到PAD
  --no_object_cost 2.0        # 可选：dummy列成本
```

## 参数调优

**no_object_weight** 控制对多余预测的惩罚强度：

| Weight | 效果 | 适用场景 |
|--------|------|----------|
| 0.05 | 轻微惩罚 | Insertion rate略高但可接受 |
| 0.1 | 中等惩罚（推荐） | 平衡insertion和deletion |
| 0.2 | 强力惩罚 | Insertion rate过高时使用 |
| 1.0 | 极强惩罚 | 极端情况，可能导致underfitting |

## 实验对比

建议进行以下对比实验：

| 实验组 | use_no_object | 预期insertion rate |
|--------|---------------|-------------------|
| Baseline | ❌ | 10.20% (已知) |
| With no object | ✅ | ~8-9% (预期降低) |

## 技术细节

**使用的token**：MT5的PAD token (`mt5_tokenizer.pad_token_id`)

**实现位置**：
- `hungarian_loss.py`: HungarianMatcher的`use_no_object`参数
- `hungarian_loss.py`: HungarianLoss的`no_object_token_id`和`no_object_weight`
- `hungarian_loss.py`: `allow_null_match`/`no_object_cost` 控制 dummy 列
- `models.py`: 自动传入PAD token ID
- `utils.py`: 命令行参数

**匹配逻辑**：
```python
if L > T and use_no_object:
    # 未匹配的预测索引
    unmatched = {0,1,2,3,4} - {0,2,4} = {1,3}
    # 标记为"no object" (tgt_idx=-1)
    indices = [(0,X), (1,∅), (2,Y), (3,∅), (4,Z)]
```

## 注意事项

1. **默认只在L > T时生效**：当预测数量≤目标数量时，所有预测都会被匹配，no object不起作用；若开启 `--allow_null_match`，即使 L == T 也允许部分预测匹配到PAD（成本由 `--no_object_cost` 控制）

2. **标签顺序**：建议保持标签真实顺序；乱序会破坏语义对齐，导致训练困难

3. **计算开销**：no object matching增加少量计算（查找未匹配索引），但可忽略不计

## 何时使用

**推荐使用**：
- ✅ Insertion rate过高（>12%）
- ✅ 打乱标签的实验
- ✅ 需要精确序列长度控制

**可以不用**：
- ❌ Insertion rate已经很低（<10%）
- ❌ 计算资源极度受限（虽然开销很小）
- ❌ 首次baseline实验（为了对比）

## 参考

基于DETR (End-to-End Object Detection with Transformers) 的Hungarian matching实现。
