# CSLR匈牙利损失 - 快速开始

## ⚡ 5分钟快速开始

### 1️⃣ 上传代码到服务器

```bash
# 在本地
cd /media/openhe/E盘/program/chatsign/Uni-Sign
git add .
git commit -m "Add Hungarian loss for CSLR task"
git push

# 在服务器
cd ~/zhewen/Uni-Sign
git pull
```

### 2️⃣ 检查依赖

```bash
# 确保已安装scipy
pip install scipy

# 检查数据集
ls ./dataset/CSL_Daily/pose_format/*.pkl | wc -l  # 应该有很多文件
ls ./dataset/CSL_Daily/label/train.json           # 应该存在
```

### 3️⃣ 运行基线实验

```bash
# 先运行baseline，建立对比基准
./script/train_cslr_baseline.sh

# 查看训练日志
tail -f out/cslr_baseline/log.txt
```

### 4️⃣ 运行匈牙利损失实验

```bash
# 建议先用0.3的权重试试
./script/train_cslr_hungarian_0.3.sh

# 查看训练日志
tail -f out/cslr_hungarian_0.3/log.txt
```

### 5️⃣ 对比结果

```bash
# 提取WER结果
echo "=== Baseline ==="
grep "Min WER" out/cslr_baseline/log.txt | tail -1

echo "=== Hungarian 0.3 ==="
grep "Min WER" out/cslr_hungarian_0.3/log.txt | tail -1
```

---

## 📋 训练检查清单

训练前确保：

- [ ] Stage 1已完成训练（`out/stage1_pretraining/best_checkpoint.pth` 存在）
- [ ] CSL_Daily数据集已下载（pose + label）
- [ ] scipy已安装
- [ ] 有足够的GPU显存（至少4张GPU）
- [ ] 网络连接正常（用于下载MT5模型，如果需要）

---

## 🔧 常用命令

### 查看训练进度
```bash
# 实时查看日志
tail -f out/cslr_hungarian_0.5/log.txt

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

### 停止训练
```bash
# 找到进程
ps aux | grep fine_tuning.py

# 杀死进程（用实际的PID替换）
kill <PID>
```

### 恢复训练（如果中断）
```bash
# 修改脚本中的checkpoint路径
# --finetune out/cslr_hungarian_0.5/checkpoint_10.pth
```

---

## 🎯 预期训练时间

**环境**: 4 x GPU, CSL_Daily数据集

- **每个epoch**: 约30-45分钟
- **总训练时间**: 约10-15小时（20 epochs）
- **加匈牙利损失**: 额外增加10-15%时间

---

## 📊 关键指标说明

训练输出示例：
```
Epoch: [15/20]
loss: 1.234
lr: 0.0003
WER: 23.45%
Del Rate: 3.2%
Ins Rate: 5.1%
Sub Rate: 15.15%
Min WER: 22.31%
```

- **WER**: 越低越好，目标<25%
- **Min WER**: 历史最佳WER，保存为best_checkpoint.pth

---

## 🚨 常见错误

### 错误1: No module named 'scipy'
```bash
pip install scipy
```

### 错误2: CUDA out of memory
```bash
# 减小batch size
# 修改脚本: --batch-size 4
```

### 错误3: FileNotFoundError: pose file
```bash
# 检查CSL_Daily数据集
ls ./dataset/CSL_Daily/pose_format/ | head
```

---

## ✅ 完成标志

训练成功完成后，应该看到：

```
out/cslr_hungarian_0.5/
├── best_checkpoint.pth      # 最佳模型
├── checkpoint_19.pth         # 最后一个epoch
├── log.txt                   # 训练日志
└── dev_tmp_pres.txt         # 预测结果（如果--eval）
```

---

## 📞 需要帮助？

如果遇到问题：
1. 检查日志：`cat out/cslr_*/log.txt | grep -i error`
2. 检查数据：`python -c "from datasets import S2T_Dataset; print('OK')"`
3. 检查模型：`python -c "from hungarian_loss import *; print('OK')"`

**祝实验成功！🎉**
