# MNIST数据选择使用说明

## 概述

这个示例展示了如何使用超参数优化框架来实现MNIST数据选择。我们的目标是根据数字6和7的验证损失，从包含0-9所有数字的训练集中自动选择最相关的样本。

## 核心思想

将数据选择转化为**样本重要性权重优化**问题：
- 为每个训练样本分配一个可学习的权重参数（超参数）
- 训练损失变成加权损失
- 验证损失在目标数字（6和7）上计算
- 通过隐式微分优化样本权重

## 使用方法

### 1. 基本使用

```python
from mnist_data_selection import MNISTDataSelector

# 创建数据选择器
selector = MNISTDataSelector(
    target_digits=[6, 7],  # 目标数字
    batch_size=64,
    device='cuda'  # 或 'cpu'
)

# 训练模型
train_losses, val_losses = selector.train(num_epochs=30)

# 分析样本重要性
top_indices, top_weights = selector.analyze_sample_importance(top_k=100)

# 可视化结果
selector.visualize_results(top_indices, num_samples=20)
```

### 2. 运行完整示例

```bash
cd hyper-opt
python mnist_data_selection.py
```

## 期望结果

1. **权重分布**: 经过训练后，样本权重应该呈现明显的差异化分布
2. **重要样本**: 权重最高的样本应该是对6和7分类最有帮助的样本
3. **可视化**: 展示的重要样本中应该包含与6、7在视觉上相似的数字

## 可能的发现

- **数字8**: 可能获得较高权重，因为它与6在视觉上相似
- **数字0**: 可能获得较高权重，因为它与6的形状有相似性
- **数字1**: 可能获得较高权重，因为它与7在形状上相似
- **数字9**: 可能获得较高权重，因为它与6的上半部分相似

## 技术细节

### DataSelectionHyperOptModel类

这是核心的数据选择模型，主要特点：

```python
class DataSelectionHyperOptModel(BaseHyperOptModel):
    def __init__(self, network, criterion, num_train_samples, initial_weight=1.0):
        # 为每个训练样本创建可学习权重
        self.sample_weights = nn.Parameter(
            torch.ones(num_train_samples) * torch.log(torch.tensor(initial_weight))
        )
```

### 超参数优化器

使用Neumann级数方法来近似求解线性系统：

```python
hyper_optimizer = NeumannHyperOptimizer(
    parameters=list(model.parameters),
    hyper_parameters=model.hyper_parameters,
    use_gauss_newton=True,
    stochastic=True
)
```

### 训练流程

1. **前向传播**: 计算加权训练损失
2. **验证损失**: 在目标数字上计算损失
3. **超参数梯度**: 通过隐式微分计算权重梯度
4. **权重更新**: 更新样本重要性权重
5. **模型更新**: 更新神经网络参数

## 自定义选项

### 修改目标数字

```python
# 选择其他数字作为目标
selector = MNISTDataSelector(target_digits=[0, 1], ...)
selector = MNISTDataSelector(target_digits=[3, 8], ...)
```

### 调整超参数

```python
# 修改学习率
hyper_optimizer = NeumannHyperOptimizer(
    ...,
    default=dict(lr=0.001),  # 降低学习率
)

# 修改Neumann迭代次数
hyper_optimizer.set_kwargs(K=50)  # 增加迭代次数
```

### 使用不同的网络架构

```python
from network import MLP

# 使用MLP替代SimpleModel
network = MLP(num_layers=3, input_shape=(28, 28), inter_dim=128)
```

## 扩展应用

这个框架可以扩展到：

1. **其他数据集**: CIFAR-10, ImageNet等
2. **多类选择**: 同时针对多个目标类别进行数据选择
3. **不平衡数据**: 自动发现和重权衡稀有样本
4. **域适应**: 选择与目标域最相关的训练样本
5. **主动学习**: 选择最有信息量的样本进行标注

## 注意事项

1. **内存使用**: 为每个样本分配权重参数会增加内存使用
2. **计算复杂度**: 隐式微分计算比标准训练更耗时
3. **收敛性**: 可能需要调整学习率和迭代次数来确保收敛
4. **初始化**: 权重初始化对最终结果有影响

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减少批大小或使用CPU
2. **收敛缓慢**: 增加学习率或迭代次数
3. **权重分化不明显**: 增加训练轮数或调整正则化

### 调试建议

1. 监控权重分布的变化
2. 检查训练和验证损失曲线
3. 可视化中间结果
4. 使用更小的数据集进行测试
