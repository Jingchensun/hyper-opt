# MNIST数据选择超参数优化

这个项目实现了基于超参数优化的MNIST数据样本选择系统，可以自动发现对特定数字识别最有帮助的训练样本。

## 🎯 核心功能

从MNIST的所有训练样本(0-9)中，自动选择对识别特定数字(默认是数字7)最有帮助的样本。

## 🚀 快速开始

### 基本使用 (推荐)
```bash
python mnist_data_selection.py --selector label_based --epochs 10 --fast_mode
```

### 更多示例
```bash
# 使用特征MLP模型
python mnist_data_selection.py --selector feature_mlp --epochs 15

# 使用CNN特征模型
python mnist_data_selection.py --selector cnn_feature --epochs 20 --fast_mode

# 完整训练 (较慢但更准确)
python mnist_data_selection.py --selector label_based --epochs 30

# 自定义参数
python mnist_data_selection.py --selector label_based --epochs 10 --batch_size 128 --top_k 200 --target_digit 9
```

## 📊 模型类型对比

| 模型类型 | 参数数量 | 训练速度 | 精细度 | 推荐度 | 说明 |
|---------|---------|----------|--------|--------|------|
| `label_based` | 10个 | ⭐⭐⭐⭐⭐ | 粗粒度 | 🥇 推荐 | 每个数字类别一个权重 |
| `feature_mlp` | ~500个 | ⭐⭐⭐⭐ | 中等 | 🥈 | 基于统计特征的MLP |
| `cnn_feature` | ~2000个 | ⭐⭐⭐ | 细粒度 | 🥉 | 基于CNN特征学习 |
| `original` | 60,000个 | ⭐ | 最细 | ❌ 不推荐 | 每样本独立权重 |

## 🛠️ 命令行参数

### 核心参数
- `--selector`: 选择模型类型 (`label_based`/`feature_mlp`/`cnn_feature`/`original`)
- `--epochs`: 训练轮数 (默认: 20)
- `--fast_mode`: 启用快速训练模式
- `--target_digit`: 目标数字 (默认: 7)

### 训练参数
- `--batch_size`: 批次大小 (默认: 64)
- `--val_subset_size`: 快速模式验证集大小 (默认: 2000)
- `--hyper_opt_freq`: 超参数优化频率 (默认: 10)

### 分析参数
- `--top_k`: 分析top-k重要样本 (默认: 100)
- `--vis_samples`: 可视化样本数量 (默认: 20)

### 其他参数
- `--device`: 设备选择 (`auto`/`cpu`/`cuda`)

## 📈 输出结果

1. **训练过程监控**: 实时显示损失变化和权重统计
2. **样本重要性排序**: 按重要性对所有训练样本排序
3. **可视化展示**: 显示最重要的样本图像
4. **统计分析**: 
   - 各数字类别的权重分布
   - 训练性能统计
   - 权重分布直方图

## 💡 使用建议

### 初学者
```bash
python mnist_data_selection.py --selector label_based --fast_mode --epochs 5
```

### 追求精度
```bash
python mnist_data_selection.py --selector feature_mlp --epochs 25
```

### 研究用途
```bash
python mnist_data_selection.py --selector cnn_feature --epochs 30
```

## 🔧 技术原理

### 双层优化框架
- **外层**: 优化样本权重λ，最小化验证损失
- **内层**: 优化模型参数θ，最小化加权训练损失

### 数学表示
```
min_λ L_val(θ*(λ))
s.t. θ*(λ) = argmin_θ Σᵢ λᵢ * L_train(xᵢ, yᵢ; θ)
```

### 优化算法
使用Neumann级数方法进行高效的超参数梯度计算，避免直接计算Hessian逆矩阵。

## 📂 文件结构
```
hyper-opt/
├── mnist_data_selection.py  # 主程序
├── model.py                 # 数据选择模型定义
├── hyper_opt.py            # 超参数优化器
├── network.py              # 基础网络结构
└── README.md               # 说明文档
```

## 🎨 可视化输出

程序会生成以下可视化结果：
1. 训练损失和验证损失曲线
2. 样本权重分布直方图  
3. 最重要样本的图像网格
4. 各数字类别的重要性统计

## ⚠️ 注意事项

1. **首次运行**: 会自动下载MNIST数据集
2. **内存需求**: CNN特征模型需要更多内存
3. **训练时间**: 完整模式比快速模式慢5-10倍
4. **设备要求**: 推荐使用GPU加速训练

## 🤝 扩展使用

可以轻松修改代码以适用于其他数据集和任务：
- 更改目标数字: `--target_digit 9`
- 适配其他数据集: 修改数据加载部分
- 自定义网络结构: 修改`network.py`
- 添加新的选择策略: 扩展`model.py`