"""
MNIST数据选择示例：
根据数字6和7的验证损失，从0-9的训练样本中选择最相关的样本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from model import DataSelectionHyperOptModel
from hyper_opt import NeumannHyperOptimizer, FixedPointHyperOptimizer
from network import SimpleModel


class IndexedDataset(Dataset):
    """带索引的数据集，用于跟踪样本在原始训练集中的位置"""
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data, target = self.base_dataset[real_idx]
        return data, target, idx  # 返回在train_indices中的位置


class MNISTDataSelector:
    def __init__(self, target_digits=[6, 7], batch_size=64, device='cpu'):
        self.target_digits = target_digits
        self.batch_size = batch_size
        self.device = device
        
        # 加载MNIST数据
        self.train_loader, self.val_loader, self.train_indices = self._load_data()
        
        # 创建模型
        self.network = SimpleModel().to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # 注意这里需要reduction='none'
        
        # 创建数据选择模型
        num_train_samples = len(self.train_indices)
        self.model = DataSelectionHyperOptModel(
            self.network, 
            self.criterion, 
            num_train_samples
        ).to(device)
        
        # 创建超参数优化器
        self.hyper_optimizer = NeumannHyperOptimizer(
            parameters=list(self.model.parameters),
            hyper_parameters=self.model.hyper_parameters,
            base_optimizer='SGD',
            default=dict(lr=0.01),
            use_gauss_newton=True,
            stochastic=True
        )
        # 设置较少的迭代次数以提高速度
        self.hyper_optimizer.set_kwargs(inner_lr=0.1, K=5)  # 默认K=20，现在设为5
        
        # 创建模型参数优化器
        self.model_optimizer = torch.optim.SGD(self.model.parameters, lr=0.01)
        
    def _load_data(self):
        """加载和预处理MNIST数据"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 加载完整数据集
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
        
        # 分离验证集（只包含目标数字6和7）
        val_indices = []
        train_indices = []
        
        for idx, (data, target) in enumerate(train_dataset):
            if target in self.target_digits:
                val_indices.append(idx)
            else:
                train_indices.append(idx)
        
        # 创建数据加载器
        train_indexed_dataset = IndexedDataset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        
        train_loader = DataLoader(train_indexed_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        
        print(f"训练样本数量: {len(train_indices)}")
        print(f"验证样本数量 (数字{self.target_digits}): {len(val_indices)}")
        
        return train_loader, val_loader, train_indices
    
    def train_step(self, train_data, train_targets, train_batch_indices):
        """执行一步训练"""
        import time
        
        step_start = time.time()
        
        # 定义训练损失函数
        def train_loss_func():
            # 重新计算训练损失（用于随机模式）
            train_loss, train_logit = self.model.train_loss(
                train_data, train_targets, train_batch_indices
            )
            return train_loss, train_logit
        
        # 计算验证损失 (这通常是最慢的部分)
        val_start = time.time()
        val_loss = self._compute_validation_loss()
        val_time = time.time() - val_start
        
        # 超参数优化步骤
        hyper_start = time.time()
        self.hyper_optimizer.step(train_loss_func, val_loss)
        hyper_time = time.time() - hyper_start
        
        # 模型参数优化步骤
        model_start = time.time()
        self.model_optimizer.zero_grad()
        train_loss, _ = train_loss_func()
        train_loss.backward()
        self.model_optimizer.step()
        model_time = time.time() - model_start
        
        total_time = time.time() - step_start
        
        # 存储时间统计用于分析
        if not hasattr(self, 'timing_stats'):
            self.timing_stats = {'val': [], 'hyper': [], 'model': [], 'total': []}
        
        self.timing_stats['val'].append(val_time)
        self.timing_stats['hyper'].append(hyper_time)
        self.timing_stats['model'].append(model_time)
        self.timing_stats['total'].append(total_time)
        
        return train_loss.item(), val_loss.item(), {
            'val_time': val_time,
            'hyper_time': hyper_time, 
            'model_time': model_time,
            'total_time': total_time
        }
    
    def _compute_validation_loss(self):
        """计算验证损失"""
        total_loss = 0
        total_samples = 0
        
        # 不使用no_grad()，因为我们需要计算验证损失的梯度
        for val_data, val_targets in self.val_loader:
            val_data, val_targets = val_data.to(self.device), val_targets.to(self.device)
            val_loss = self.model.validation_loss(val_data, val_targets)
            total_loss += val_loss * val_data.size(0)  # 直接使用tensor，不转换为item
            total_samples += val_data.size(0)
        
        return total_loss / total_samples
    
    def train(self, num_epochs=50):
        """训练数据选择模型"""
        train_losses = []
        val_losses = []
        
        print("开始训练数据选择模型...")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            epoch_train_loss = 0
            epoch_val_loss = 0
            num_batches = 0
            
            # 每个epoch开始时显示样本权重统计
            if epoch % 5 == 0:
                weights = self.model.get_sample_weights().detach().cpu().numpy()
                print(f"\nEpoch {epoch} - 样本权重统计:")
                print(f"  最大权重: {weights.max():.6f}")
                print(f"  最小权重: {weights.min():.6f}")
                print(f"  平均权重: {weights.mean():.6f}")
                print(f"  权重标准差: {weights.std():.6f}")
                print(f"  高权重样本数 (>avg): {(weights > weights.mean()).sum()}")
            
            for batch_idx, (train_data, train_targets, train_batch_indices) in enumerate(self.train_loader):
                train_data, train_targets = train_data.to(self.device), train_targets.to(self.device)
                # train_batch_indices 已经从 IndexedDataset 中获得
                
                # 执行训练步骤
                train_loss, val_loss, timing = self.train_step(
                    train_data, train_targets, train_batch_indices
                )
                
                epoch_train_loss += train_loss
                epoch_val_loss += val_loss
                num_batches += 1
                
                # 每20个batch打印一次进度
                if batch_idx % 20 == 0:
                    current_weights = self.model.get_sample_weights()[train_batch_indices].detach().cpu().numpy()
                    print(f"  Epoch {epoch:2d}, Batch {batch_idx:3d}/{len(self.train_loader):3d} | "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                          f"当前batch权重: {current_weights.mean():.4f}±{current_weights.std():.4f}")
                    print(f"    ⏱️  时间分析: 验证损失 {timing['val_time']:.2f}s, "
                          f"超参数优化 {timing['hyper_time']:.2f}s, "
                          f"模型优化 {timing['model_time']:.2f}s, "
                          f"总计 {timing['total_time']:.2f}s")
            
            avg_train_loss = epoch_train_loss / num_batches
            avg_val_loss = epoch_val_loss / num_batches
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # 每个epoch结束时打印总结
            print(f"\n>>> Epoch {epoch:2d} 完成 | "
                  f"平均训练损失: {avg_train_loss:.4f}, 平均验证损失: {avg_val_loss:.4f}")
            
            # 显示损失变化趋势
            if epoch > 0:
                train_change = avg_train_loss - train_losses[-2]
                val_change = avg_val_loss - val_losses[-2]
                train_trend = "↓" if train_change < 0 else "↑" if train_change > 0 else "="
                val_trend = "↓" if val_change < 0 else "↑" if val_change > 0 else "="
                print(f"    损失变化: 训练 {train_trend} {train_change:+.4f}, 验证 {val_trend} {val_change:+.4f}")
            
            # 显示这个epoch的平均时间统计
            if hasattr(self, 'timing_stats'):
                import numpy as np
                recent_stats = {}
                for key in self.timing_stats:
                    recent_stats[key] = np.mean(self.timing_stats[key][-num_batches:])
                
                print(f"    ⏱️  平均时间分析: 验证损失 {recent_stats['val']:.2f}s ({recent_stats['val']/recent_stats['total']*100:.1f}%), "
                      f"超参数优化 {recent_stats['hyper']:.2f}s ({recent_stats['hyper']/recent_stats['total']*100:.1f}%), "
                      f"模型优化 {recent_stats['model']:.2f}s ({recent_stats['model']/recent_stats['total']*100:.1f}%)")
                print(f"    📊 每batch平均时间: {recent_stats['total']:.2f}s, 预计完成时间: {recent_stats['total'] * len(self.train_loader) * (num_epochs - epoch - 1) / 60:.1f} 分钟")
            
            print("-" * 80)
        
        return train_losses, val_losses
    
    def train_fast(self, num_epochs=50, val_subset_size=1000, hyper_opt_freq=5):
        """快速训练版本 - 优化性能"""
        print("🚀 使用快速训练模式 (性能优化版本)")
        print(f"   - 验证集子集大小: {val_subset_size}")
        print(f"   - 超参数优化频率: 每 {hyper_opt_freq} 个batch")
        
        # 创建验证集子集
        val_subset_indices = torch.randperm(len(self.val_loader.dataset))[:val_subset_size]
        
        train_losses = []
        val_losses = []
        
        print("=" * 80)
        
        for epoch in range(num_epochs):
            epoch_train_loss = 0
            epoch_val_loss = 0
            num_batches = 0
            
            # 每个epoch开始时显示样本权重统计
            if epoch % 5 == 0:
                weights = self.model.get_sample_weights().detach().cpu().numpy()
                print(f"\nEpoch {epoch} - 样本权重统计:")
                print(f"  最大权重: {weights.max():.6f}")
                print(f"  最小权重: {weights.min():.6f}")
                print(f"  平均权重: {weights.mean():.6f}")
                print(f"  权重标准差: {weights.std():.6f}")
                print(f"  高权重样本数 (>avg): {(weights > weights.mean()).sum()}")
            
            for batch_idx, (train_data, train_targets, train_batch_indices) in enumerate(self.train_loader):
                train_data, train_targets = train_data.to(self.device), train_targets.to(self.device)
                
                # 只在指定频率下进行超参数优化
                if batch_idx % hyper_opt_freq == 0:
                    train_loss, val_loss, timing = self.train_step_fast(
                        train_data, train_targets, train_batch_indices, val_subset_indices
                    )
                else:
                    # 只进行模型参数优化
                    train_loss, val_loss, timing = self.train_step_model_only(
                        train_data, train_targets, train_batch_indices
                    )
                
                epoch_train_loss += train_loss
                epoch_val_loss += val_loss
                num_batches += 1
                
                # 每50个batch打印一次进度
                if batch_idx % 50 == 0:
                    current_weights = self.model.get_sample_weights()[train_batch_indices].detach().cpu().numpy()
                    print(f"  Epoch {epoch:2d}, Batch {batch_idx:3d}/{len(self.train_loader):3d} | "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                          f"当前batch权重: {current_weights.mean():.4f}±{current_weights.std():.4f}")
                    if timing:
                        print(f"    ⏱️  时间: {timing['total_time']:.2f}s")
            
            avg_train_loss = epoch_train_loss / num_batches
            avg_val_loss = epoch_val_loss / num_batches
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # 每个epoch结束时打印总结
            print(f"\n>>> Epoch {epoch:2d} 完成 | "
                  f"平均训练损失: {avg_train_loss:.4f}, 平均验证损失: {avg_val_loss:.4f}")
            
            if epoch > 0:
                train_change = avg_train_loss - train_losses[-2]
                val_change = avg_val_loss - val_losses[-2]
                train_trend = "↓" if train_change < 0 else "↑" if train_change > 0 else "="
                val_trend = "↓" if val_change < 0 else "↑" if val_change > 0 else "="
                print(f"    损失变化: 训练 {train_trend} {train_change:+.4f}, 验证 {val_trend} {val_change:+.4f}")
            
            print("-" * 80)
        
        return train_losses, val_losses
    
    def train_step_fast(self, train_data, train_targets, train_batch_indices, val_subset_indices):
        """快速训练步骤 - 使用验证集子集"""
        import time
        
        step_start = time.time()
        
        # 定义训练损失函数
        def train_loss_func():
            train_loss, train_logit = self.model.train_loss(
                train_data, train_targets, train_batch_indices
            )
            return train_loss, train_logit
        
        # 计算验证损失 (使用子集)
        val_start = time.time()
        val_loss = self._compute_validation_loss_subset(val_subset_indices)
        val_time = time.time() - val_start
        
        # 超参数优化步骤
        hyper_start = time.time()
        self.hyper_optimizer.step(train_loss_func, val_loss)
        hyper_time = time.time() - hyper_start
        
        # 模型参数优化步骤
        model_start = time.time()
        self.model_optimizer.zero_grad()
        train_loss, _ = train_loss_func()
        train_loss.backward()
        self.model_optimizer.step()
        model_time = time.time() - model_start
        
        total_time = time.time() - step_start
        
        return train_loss.item(), val_loss.item(), {
            'val_time': val_time,
            'hyper_time': hyper_time, 
            'model_time': model_time,
            'total_time': total_time
        }
    
    def train_step_model_only(self, train_data, train_targets, train_batch_indices):
        """只更新模型参数的训练步骤"""
        import time
        
        step_start = time.time()
        
        # 只进行模型参数优化
        self.model_optimizer.zero_grad()
        train_loss, _ = self.model.train_loss(train_data, train_targets, train_batch_indices)
        train_loss.backward()
        self.model_optimizer.step()
        
        total_time = time.time() - step_start
        
        # 使用上次的验证损失作为近似 (或者可以设为0)
        return train_loss.item(), 0.0, {'total_time': total_time}
    
    def _compute_validation_loss_subset(self, subset_indices):
        """计算验证集子集的损失"""
        total_loss = 0
        total_samples = 0
        
        # 从验证集中采样子集
        val_dataset = self.val_loader.dataset
        subset_data = []
        subset_targets = []
        
        for idx in subset_indices:
            data, target = val_dataset[idx]
            subset_data.append(data)
            subset_targets.append(target)
        
        if len(subset_data) > 0:
            val_data = torch.stack(subset_data).to(self.device)
            val_targets = torch.tensor(subset_targets).to(self.device)
            val_loss = self.model.validation_loss(val_data, val_targets)
            return val_loss
        else:
            return torch.tensor(0.0, device=self.device)
    
    def analyze_sample_importance(self, top_k=100):
        """分析样本重要性并返回最重要的样本"""
        
        # 获取所有样本的权重
        weights = self.model.get_sample_weights().detach().cpu().numpy()
        
        # 找到权重最高的样本
        top_indices = np.argsort(weights)[-top_k:][::-1]
        top_weights = weights[top_indices]
        
        print(f"\n{'='*60}")
        print(f"样本重要性分析结果 (Top {top_k})")
        print(f"{'='*60}")
        print(f"🎯 最高权重样本: {top_weights[0]:.6f}")
        print(f"📊 权重统计:")
        print(f"   - 最低权重: {weights.min():.6f}")
        print(f"   - 平均权重: {weights.mean():.6f}")
        print(f"   - 权重标准差: {weights.std():.6f}")
        print(f"   - 权重范围: [{weights.min():.6f}, {weights.max():.6f}]")
        
        # 权重分布分析
        high_weight_count = (weights > weights.mean() + weights.std()).sum()
        low_weight_count = (weights < weights.mean() - weights.std()).sum()
        print(f"📈 权重分布:")
        print(f"   - 高权重样本 (>μ+σ): {high_weight_count} ({high_weight_count/len(weights)*100:.1f}%)")
        print(f"   - 低权重样本 (<μ-σ): {low_weight_count} ({low_weight_count/len(weights)*100:.1f}%)")
        print(f"   - 中等权重样本: {len(weights) - high_weight_count - low_weight_count}")
        
        # Top样本的权重分布
        print(f"🏆 Top {top_k} 样本权重范围: [{top_weights[-1]:.6f}, {top_weights[0]:.6f}]")
        print(f"   - Top 10 权重平均: {top_weights[:10].mean():.6f}")
        print(f"   - Top 50 权重平均: {top_weights[:50].mean():.6f}")
        
        return top_indices, top_weights
    
    def visualize_results(self, top_indices, num_samples=20):
        """可视化最重要的样本"""
        
        # 加载原始数据集用于可视化
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST('data', train=True, transform=transform)
        
        # 分析top样本的数字分布
        digit_counts = {}
        sample_weights = self.model.get_sample_weights().detach().cpu().numpy()
        
        for i in range(min(num_samples, len(top_indices))):
            real_idx = self.train_indices[top_indices[i]]
            _, label = dataset[real_idx]
            label = int(label)
            if label not in digit_counts:
                digit_counts[label] = 0
            digit_counts[label] += 1
        
        print(f"🔍 Top {num_samples} 样本的数字分布:")
        for digit in sorted(digit_counts.keys()):
            percentage = (digit_counts[digit] / num_samples) * 100
            print(f"   数字 {digit}: {digit_counts[digit]:2d} 个样本 ({percentage:4.1f}%)")
        
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        fig.suptitle(f'前{num_samples}个最重要的训练样本\n(基于数字6&7的验证损失优化)', fontsize=16)
        
        for i in range(min(num_samples, len(top_indices))):
            row = i // 5
            col = i % 5
            
            # 获取真实的数据集索引
            real_idx = self.train_indices[top_indices[i]]
            image, label = dataset[real_idx]
            weight = sample_weights[top_indices[i]]
            
            axes[row, col].imshow(image.squeeze(), cmap='gray')
            axes[row, col].set_title(f'数字: {label}\n权重: {weight:.4f}', fontsize=10)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return digit_counts


def main():
    """主函数：演示MNIST数据选择"""
    
    print("🎯 MNIST数据选择超参数优化")
    print("=" * 60)
    print("📊 性能分析说明:")
    print("   - 超参数优化比普通训练慢10-50倍")
    print("   - 主要瓶颈: 验证损失计算 (~60-80%) + 二阶优化 (~20-30%)")
    print("   - 快速模式可提速5-10倍，略微降低精度")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")
    
    # 创建数据选择器
    print("📦 正在初始化数据选择器...")
    selector = MNISTDataSelector(target_digits=[6, 7], batch_size=64, device=device)
    
    print(f"📊 模型参数数量: {sum(p.numel() for p in selector.model.parameters):,}")
    print(f"🎯 超参数数量: {sum(p.numel() for p in selector.model.hyper_parameters):,}")
    print(f"⚙️  训练批次大小: {selector.batch_size}")
    print(f"🔄 每个epoch批次数: {len(selector.train_loader)}")
    
    # 显示初始权重统计
    initial_weights = selector.model.get_sample_weights().detach().cpu().numpy()
    print(f"\n📈 初始样本权重统计:")
    print(f"   平均权重: {initial_weights.mean():.6f}")
    print(f"   权重标准差: {initial_weights.std():.6f}")
    print(f"   权重范围: [{initial_weights.min():.6f}, {initial_weights.max():.6f}]")
    
    # 训练模型
    import time
    start_time = time.time()
    print(f"\n⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 选择训练模式
    print(f"\n🔧 训练配置:")
    print(f"   1. 完整训练: 每个batch都进行超参数优化 (准确但慢)")
    print(f"   2. 快速训练: 减少验证集大小和超参数优化频率 (快速但可能稍差)")
    
    # 这里默认使用快速训练，你可以改为 selector.train(num_epochs=30) 来使用完整训练
    USE_FAST_TRAINING = True
    
    if USE_FAST_TRAINING:
        train_losses, val_losses = selector.train_fast(
            num_epochs=30, 
            val_subset_size=2000,  # 从12183个样本减少到2000个
            hyper_opt_freq=10      # 每10个batch才进行一次超参数优化
        )
    else:
        train_losses, val_losses = selector.train(num_epochs=30)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n✅ 训练完成!")
    print(f"⏱️  总训练时间: {training_time:.1f} 秒 ({training_time/60:.1f} 分钟)")
    print(f"📉 最终训练损失: {train_losses[-1]:.4f}")
    print(f"📊 最终验证损失: {val_losses[-1]:.4f}")
    
    # 性能分析总结
    if hasattr(selector, 'timing_stats') and selector.timing_stats['total']:
        import numpy as np
        avg_batch_time = np.mean(selector.timing_stats['total'])
        avg_val_time = np.mean(selector.timing_stats['val'])
        avg_hyper_time = np.mean(selector.timing_stats['hyper'])
        
        print(f"\n📈 性能分析总结:")
        print(f"   平均每batch时间: {avg_batch_time:.2f}s")
        print(f"   验证损失计算: {avg_val_time:.2f}s ({avg_val_time/avg_batch_time*100:.1f}%)")
        print(f"   超参数优化: {avg_hyper_time:.2f}s ({avg_hyper_time/avg_batch_time*100:.1f}%)")
        
        estimated_fast_time = training_time * 0.1  # 快速模式估计提速10倍
        print(f"   估计快速模式时间: {estimated_fast_time/60:.1f} 分钟 (提速约10倍)")
    
    print("\n💡 进一步优化建议:")
    print("   1. 减少验证集大小: val_subset_size=500-1000")
    print("   2. 降低超参数优化频率: hyper_opt_freq=20-50")  
    print("   3. 减少Neumann迭代次数: K=3-5")
    print("   4. 使用更简单的优化器: use_gauss_newton=False")
    
    # 分析样本重要性
    top_indices, top_weights = selector.analyze_sample_importance(top_k=100)
    
    # 可视化结果
    print(f"\n🎨 正在生成可视化结果...")
    selector.visualize_results(top_indices, num_samples=20)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失 (6&7)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练过程')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(selector.model.get_sample_weights().detach().cpu().numpy(), bins=50, alpha=0.7)
    plt.xlabel('样本权重')
    plt.ylabel('频数')
    plt.title('样本权重分布')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
