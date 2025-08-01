"""
MNIST数据选择示例：
根据数字7的验证损失，从0-9的所有训练样本中选择最相关的样本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import json
import os

from model import (LabelBasedDataSelectionModel, 
                   FeatureMlpDataSelectionModel, CnnFeatureDataSelectionModel)
from hyper_opt import NeumannHyperOptimizer, FixedPointHyperOptimizer
from network import SimpleModel


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MNIST数据选择超参数优化')
    
    # 模型选择参数
    parser.add_argument('--selector', type=str, default='label_based',
                        choices=['label_based', 'feature_mlp', 'cnn_feature'],
                        help='选择数据选择模型类型')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=2,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--val_subset_size', type=int, default=2000,
                        help='验证集子集大小')
    parser.add_argument('--hyper_opt_freq', type=int, default=10,
                        help='超参数优化频率')
    
    # 分析参数
    parser.add_argument('--top_k', type=int, default=1000,
                        help='分析top-k重要样本')
    parser.add_argument('--vis_samples', type=int, default=20,
                        help='可视化样本数量')
    
    # 其他参数
    parser.add_argument('--target_digit', type=int, default=7,
                        help='目标数字')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备选择 (auto/cpu/cuda)')
    
    return parser.parse_args()


def print_model_info(model_type):
    """打印模型类型信息"""
    model_info = {
        'label_based': {
            'name': '基于标签的模型',
            'params': '10个',
            'pros': '最简单，最快，易解释',
            'cons': '粒度较粗',
            'speed': '⭐⭐⭐⭐⭐',
            'recommend': '🥇 推荐'
        },
        'feature_mlp': {
            'name': '基于特征MLP的模型',
            'params': '~500个',
            'pros': '特征动态分配，较精细',
            'cons': '需手工特征设计',
            'speed': '⭐⭐⭐⭐',
            'recommend': '🥈'
        },
        'cnn_feature': {
            'name': '基于CNN特征的模型',
            'params': '~2000个',
            'pros': '自动学习特征，最灵活',
            'cons': '参数最多，训练最慢',
            'speed': '⭐⭐⭐',
            'recommend': '🥉'
        }
    }
    
    info = model_info[model_type]
    print(f"🧠 选择的模型: {info['name']}")
    print(f"   📊 参数数量: {info['params']}")
    print(f"   ✅ 优点: {info['pros']}")
    print(f"   ⚠️  缺点: {info['cons']}")
    print(f"   🚀 训练速度: {info['speed']}")
    print(f"   🏆 推荐度: {info['recommend']}")


def setup_device(device_arg):
    """设置计算设备"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"🚀 使用设备: {device}")
    return device


def print_training_stats(selector, model_type):
    """打印训练统计信息"""
    print(f"📊 模型参数数量: {sum(p.numel() for p in selector.model.parameters):,}")
    print(f"🎯 超参数数量: {sum(p.numel() for p in selector.model.hyper_parameters):,}")
    print(f"⚙️  训练批次大小: {selector.batch_size}")
    print(f"🔄 每个epoch批次数: {len(selector.train_loader)}")
    
    # 显示初始权重统计
    print_initial_weights(selector, model_type)


def print_initial_weights(selector, model_type):
    """打印初始权重统计"""
    if model_type == 'label_based':
        initial_class_weights = selector.model.get_class_weights().detach().cpu().numpy()
        print(f"\n📈 初始类别权重统计:")
        for i, weight in enumerate(initial_class_weights):
            print(f"   数字 {i}: {weight:.6f}")
        print(f"   平均权重: {initial_class_weights.mean():.6f}")
        print(f"   权重范围: [{initial_class_weights.min():.6f}, {initial_class_weights.max():.6f}]")
    else:
        print(f"\n📈 初始权重网络参数统计:")
        total_params = sum(p.numel() for p in selector.model.hyper_parameters)
        print(f"   超参数总数: {total_params}")
        all_params = torch.cat([p.flatten() for p in selector.model.hyper_parameters])
        print(f"   参数范围: [{all_params.min():.6f}, {all_params.max():.6f}]")
        print(f"   参数均值: {all_params.mean():.6f}")
        print(f"   参数标准差: {all_params.std():.6f}")


def analyze_and_visualize(selector, args):
    """分析样本重要性并可视化"""
    print(f"\n📊 分析样本重要性...")
    top_indices, top_weights, weights_stats, digit_distribution = selector.analyze_sample_importance(
        top_k=args.top_k
    )
    
    print(f"\n🎨 生成可视化结果...")
    # 为可视化准备权重数组
    all_weights = prepare_weights_for_visualization(selector, args.selector)
    selector.visualize_results(top_indices, all_weights, num_samples=args.vis_samples)
    
    return all_weights, weights_stats, digit_distribution


def prepare_weights_for_visualization(selector, model_type):
    """为可视化准备完整的权重数组"""
    if model_type == 'label_based':
        class_weights = selector.model.get_class_weights().detach().cpu().numpy()
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST('data', train=True, transform=transform)
        all_weights = []
        for train_idx in selector.train_indices:
            _, label = dataset[train_idx]
            all_weights.append(class_weights[label])
        return np.array(all_weights)
    
    else:  # feature_mlp 或 cnn_feature
        all_weights = []
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('data', train=True, transform=transform)
        
        batch_size = 256
        for i in range(0, len(selector.train_indices), batch_size):
            batch_indices = selector.train_indices[i:i+batch_size]
            batch_data = []
            
            for idx in batch_indices:
                data, _ = dataset[idx]
                batch_data.append(data)
            
            if len(batch_data) > 0:
                batch_tensor = torch.stack(batch_data).to(selector.device)
                with torch.no_grad():
                    batch_weights = selector.model.get_sample_weights(batch_tensor)
                    all_weights.extend(batch_weights.detach().cpu().numpy())
        
        return np.array(all_weights)


def save_visualizations(selector, train_losses, val_losses, all_weights, 
                       weights_stats, digit_distribution, args):
    """Save all visualization plots to output folder"""
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set font for better compatibility
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    print(f"💾 Saving visualization results to {output_dir} folder...")
    
    # 1. Training loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2, color='blue')
    plt.plot(val_losses, label=f'Validation Loss (Digit {args.target_digit})', linewidth=2, color='red')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'MNIST Data Selection Training Process ({args.selector})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_loss_{args.selector}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sample weight distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Sample Weight', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Sample Weight Distribution ({args.selector})', fontsize=14)
    plt.axvline(all_weights.mean(), color='red', linestyle='--', 
                label=f'Mean: {all_weights.mean():.4f}')
    plt.axvline(all_weights.mean() + all_weights.std(), color='orange', linestyle='--', 
                label=f'μ+σ: {all_weights.mean() + all_weights.std():.4f}')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'weight_distribution_{args.selector}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Weight statistics by digit class
    if args.selector == 'label_based':
        class_weights = selector.model.get_class_weights().detach().cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(10), class_weights, color='lightcoral', alpha=0.8, edgecolor='black')
        plt.xlabel('Digit Class', fontsize=12)
        plt.ylabel('Class Weight', fontsize=12)
        plt.title(f'Weight by Digit Class (Target: Digit {args.target_digit})', fontsize=14)
        plt.xticks(range(10))
        
        # Highlight target digit
        bars[args.target_digit].set_color('gold')
        bars[args.target_digit].set_edgecolor('red')
        bars[args.target_digit].set_linewidth(3)
        
        # Add value labels
        for i, weight in enumerate(class_weights):
            plt.text(i, weight + 0.01, f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'weights_by_digit_{args.selector}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Top samples distribution statistics
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    digits = list(range(10))
    counts = [digit_distribution.get(d, 0) for d in digits]
    percentages = [c / args.top_k * 100 for c in counts]
    
    # Create bar chart
    colors = ['gold' if d == args.target_digit else 'lightblue' for d in digits]
    bars = plt.bar(digits, percentages, color=colors, alpha=0.8, edgecolor='black')
    
    plt.xlabel('Digit Class', fontsize=12)
    plt.ylabel('Percentage in Top Samples (%)', fontsize=12)
    plt.title(f'Top {args.top_k} Important Samples Distribution\n(Most Helpful Samples for Digit {args.target_digit} Recognition)', fontsize=14)
    plt.xticks(digits)
    
    # Add value labels
    for i, (count, pct) in enumerate(zip(counts, percentages)):
        if count > 0:
            plt.text(i, pct + 1, f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
    
    # Add baseline (random distribution should be 10%)
    plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, 
                label='Random Distribution Baseline (10%)')
    
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(percentages) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'top_samples_distribution_{args.selector}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Generate summary statistics plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: Training curves
    ax1.plot(train_losses, label='Training Loss', linewidth=2, color='blue')
    ax1.plot(val_losses, label='Validation Loss', linewidth=2, color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Process')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Weight distribution
    ax2.hist(all_weights, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(all_weights.mean(), color='red', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Sample Weight')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Sample Weight Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Class weights (if label_based) or weight statistics
    if args.selector == 'label_based':
        ax3.bar(range(10), class_weights, color='lightcoral', alpha=0.8)
        ax3.set_xlabel('Digit Class')
        ax3.set_ylabel('Class Weight')
        ax3.set_title('Weight by Digit Class')
        ax3.set_xticks(range(10))
    else:
        # Show weight statistics
        stats_text = f"""Weight Statistics:
        Min: {weights_stats['min_weight']:.4f}
        Max: {weights_stats['max_weight']:.4f}
        Mean: {weights_stats['mean_weight']:.4f}
        Std: {weights_stats['std_weight']:.4f}
        High Weight Samples: {weights_stats['high_weight_count']}
        Low Weight Samples: {weights_stats['low_weight_count']}"""
        ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax3.set_title('Weight Statistics')
        ax3.axis('off')
    
    # Subplot 4: Top samples distribution
    ax4.bar(digits, percentages, color=colors, alpha=0.8)
    ax4.axhline(y=10, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Digit Class')
    ax4.set_ylabel('Percentage (%)')
    ax4.set_title(f'Top {args.top_k} Samples Distribution')
    ax4.set_xticks(digits)
    
    plt.suptitle(f'MNIST Data Selection Analysis Results ({args.selector}, Target: Digit {args.target_digit})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'analysis_summary_{args.selector}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved 5 visualization plots to {output_dir} folder:")
    print(f"   📈 training_loss_{args.selector}.png")
    print(f"   📊 weight_distribution_{args.selector}.png") 
    if args.selector == 'label_based':
        print(f"   🔢 weights_by_digit_{args.selector}.png")
    print(f"   🏆 top_samples_distribution_{args.selector}.png")
    print(f"   📋 analysis_summary_{args.selector}.png")


def plot_results(train_losses, val_losses, all_weights):
    """Plot training results (display functionality)"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss (Digit 7)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Process')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(all_weights, bins=50, alpha=0.7)
    plt.xlabel('Sample Weight')
    plt.ylabel('Frequency')
    plt.title('Sample Weight Distribution')
    
    plt.tight_layout()
    plt.show()


def print_performance_summary(start_time, train_losses, val_losses):
    """打印性能总结"""
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ 训练完成!")
    print(f"⏱️  总训练时间: {training_time:.1f} 秒 ({training_time/60:.1f} 分钟)")
    print(f"📉 最终训练损失: {train_losses[-1]:.4f}")
    print(f"📊 最终验证损失: {val_losses[-1]:.4f}")


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
    def __init__(self, target_digits=[7], batch_size=64, device='cpu', model_type='label_based'):
        self.target_digits = target_digits
        self.batch_size = batch_size
        self.device = device
        self.model_type = model_type
        
        # 加载MNIST数据
        self.train_loader, self.val_loader, self.train_indices = self._load_data()
        
        # 创建模型
        self.network = SimpleModel().to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # 注意这里需要reduction='none'
        
        # 根据模型类型创建数据选择模型
        num_train_samples = len(self.train_indices)
        self.model = self._create_selection_model(num_train_samples).to(device)
        
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
        
        # 分离验证集（只包含目标数字7）
        val_indices = []
        train_indices = []
        
        for idx, (data, target) in enumerate(train_dataset):
            # 所有样本都加入训练集
            train_indices.append(idx)
            # 只有目标数字加入验证集
            if target in self.target_digits:
                val_indices.append(idx)
        
        # 创建数据加载器
        train_indexed_dataset = IndexedDataset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        
        train_loader = DataLoader(train_indexed_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        
        print(f"训练样本数量: {len(train_indices)}")
        print(f"验证样本数量 (数字{self.target_digits}): {len(val_indices)}")
        
        return train_loader, val_loader, train_indices
    
    def _create_selection_model(self, num_train_samples):
        """根据模型类型创建相应的数据选择模型"""
        if self.model_type == 'label_based':
            # 基于标签的简化模型（推荐）
            return LabelBasedDataSelectionModel(
                self.network, 
                self.criterion,
                num_classes=10
            )
        elif self.model_type == 'feature_mlp':
            # 基于特征MLP的模型
            return FeatureMlpDataSelectionModel(
                self.network, 
                self.criterion,
                feature_dim=16,
                hidden_dim=32
            )
        elif self.model_type == 'cnn_feature':
            # 基于CNN特征的模型
            return CnnFeatureDataSelectionModel(
                self.network, 
                self.criterion,
                hidden_dim=64
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. "
                           f"Choose from: 'label_based', 'feature_mlp', 'cnn_feature'")
    
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
    
    def train_fast(self, num_epochs=10, val_subset_size=1000, hyper_opt_freq=5):
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
            
            # 每个epoch开始时显示权重统计
            if epoch % 5 == 0:
                if self.model_type == 'label_based':
                    class_weights = self.model.get_class_weights().detach().cpu().numpy()
                    print(f"\nEpoch {epoch} - 类别权重统计:")
                    for i, weight in enumerate(class_weights):
                        print(f"  数字 {i}: {weight:.6f}")
                    print(f"  最大类别权重: {class_weights.max():.6f}")
                    print(f"  最小类别权重: {class_weights.min():.6f}")
                else:
                    print(f"\nEpoch {epoch} - 权重网络参数统计:")
                    total_params = sum(p.numel() for p in self.model.hyper_parameters)
                    print(f"  超参数数量: {total_params}")
                    # 显示权重网络的参数范围
                    all_params = torch.cat([p.flatten() for p in self.model.hyper_parameters])
                    print(f"  参数范围: [{all_params.min():.6f}, {all_params.max():.6f}]")
            
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
                    if self.model_type == 'label_based':
                        batch_weights = self.model.get_sample_weights_by_labels(train_targets).detach().cpu().numpy()
                        weight_info = f"当前batch权重: {batch_weights.mean():.4f}±{batch_weights.std():.4f}"
                    else:
                        # feature_mlp or cnn_feature: 实时计算权重
                        with torch.no_grad():
                            batch_weights = self.model.get_sample_weights(train_data).detach().cpu().numpy()
                        weight_info = f"当前batch权重: {batch_weights.mean():.4f}±{batch_weights.std():.4f}"
                    
                    print(f"  Epoch {epoch:2d}, Batch {batch_idx:3d}/{len(self.train_loader):3d} | "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | {weight_info}")
                    if timing:
                        print(f"    ⏱️  时间: {timing['total_time']:.2f}s")
            
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
            
            print("-" * 80)
        
        return train_losses, val_losses
    
    def analyze_sample_importance(self, top_k=100):
        """分析样本重要性并返回最重要的样本"""
        
        # 根据模型类型获取权重
        if self.model_type == 'label_based':
            # 基于标签的模型：根据类别分配权重
            class_weights = self.model.get_class_weights().detach().cpu().numpy()
            print(f"📊 各数字类别权重:")
            for i, weight in enumerate(class_weights):
                print(f"   数字 {i}: {weight:.6f}")
            
            # 为每个训练样本分配对应类别的权重
            weights = []
            transform = transforms.Compose([transforms.ToTensor()])
            dataset = datasets.MNIST('data', train=True, transform=transform)
            
            for train_idx in self.train_indices:
                _, label = dataset[train_idx]
                weights.append(class_weights[label])
            weights = np.array(weights)
            
        elif self.model_type in ['feature_mlp', 'cnn_feature']:
            # 基于特征的模型：需要遍历所有训练样本计算权重
            print("⏳ 计算所有训练样本的权重（可能需要一些时间）...")
            weights = []
            
            # 创建数据加载器来批量处理
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset = datasets.MNIST('data', train=True, transform=transform)
            
            batch_size = 256  # 较大的batch size以提高效率
            for i in range(0, len(self.train_indices), batch_size):
                batch_indices = self.train_indices[i:i+batch_size]
                batch_data = []
                
                for idx in batch_indices:
                    data, _ = dataset[idx]
                    batch_data.append(data)
                
                if len(batch_data) > 0:
                    batch_tensor = torch.stack(batch_data).to(self.device)
                    with torch.no_grad():
                        batch_weights = self.model.get_sample_weights(batch_tensor)
                        weights.extend(batch_weights.detach().cpu().numpy())
            
            weights = np.array(weights)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
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
        
        # 分析数字分布
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST('data', train=True, transform=transform)
        digit_distribution = {}
        
        for i in range(min(top_k, len(top_indices))):
            real_idx = self.train_indices[top_indices[i]]
            _, label = dataset[real_idx]
            label = int(label)
            if label not in digit_distribution:
                digit_distribution[label] = 0
            digit_distribution[label] += 1
        
        # 创建权重统计信息
        weights_stats = {
            "min_weight": float(weights.min()),
            "max_weight": float(weights.max()),
            "mean_weight": float(weights.mean()),
            "std_weight": float(weights.std()),
            "high_weight_count": int(high_weight_count),
            "low_weight_count": int(low_weight_count),
            "total_samples": len(weights),
            "top_weights_mean": float(top_weights.mean()),
            "top_10_mean": float(top_weights[:min(10, len(top_weights))].mean()),
            "top_50_mean": float(top_weights[:min(50, len(top_weights))].mean()) if len(top_weights) >= 50 else float(top_weights.mean())
        }
        
        return top_indices, top_weights, weights_stats, digit_distribution
    
    def visualize_results(self, top_indices, weights, num_samples=20):
        """可视化最重要的样本"""
        
        # 加载原始数据集用于可视化
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST('data', train=True, transform=transform)
        
        # 分析top样本的数字分布
        digit_counts = {}
        sample_weights = weights  # 使用传入的权重
        
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
        
        # 简化的可视化 - 只显示数字分布统计
        print(f"\n📊 最重要样本的数字分布统计:")
        total_samples = sum(digit_counts.values())
        for digit in range(10):
            count = digit_counts.get(digit, 0)
            if count > 0:
                print(f"   数字 {digit}: {'█' * int(count * 20 / total_samples)} {count}/{total_samples}")
        
        return digit_counts


def main():
    """主函数：演示MNIST数据选择"""
    
    print("🎯 MNIST数据选择超参数优化 (精简版)")
    print("=" * 60)
    print("📋 任务说明:")
    print("   - 从MNIST所有训练样本(0-9)中选择对数字7识别最有帮助的样本")
    print("   - 使用双层优化：外层学习样本权重，内层训练分类器")
    print("   - 只使用快速训练模式")
    print("=" * 60)
    
    # 1. 解析参数和初始化
    args = get_args()
    print_model_info(args.selector)
    device = setup_device(args.device)
    
    # 2. 创建数据选择器
    print("\n📦 初始化数据选择器...")
    selector = MNISTDataSelector(
        target_digits=[args.target_digit], 
        batch_size=args.batch_size, 
        device=device, 
        model_type=args.selector
    )
    print_training_stats(selector, args.selector)
    
    # 3. 训练模型 (只使用快速模式)
    print(f"\n⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    train_losses, val_losses = selector.train_fast(
        num_epochs=args.epochs,
        val_subset_size=args.val_subset_size,
        hyper_opt_freq=args.hyper_opt_freq
    )
    
    print_performance_summary(start_time, train_losses, val_losses)
    
    # 4. 分析和可视化
    all_weights, weights_stats, digit_distribution = analyze_and_visualize(selector, args)
    plot_results(train_losses, val_losses, all_weights)
    
    # 5. 保存所有可视化结果
    save_visualizations(selector, train_losses, val_losses, all_weights, 
                       weights_stats, digit_distribution, args)
    
    print(f"\n🎉 实验完成！")
    print(f"💡 从训练结果可以看出哪些数字类别对识别数字7最有帮助!")


if __name__ == "__main__":
    main()


"""
📖 使用示例：

1. 基本使用 (推荐的label_based模型):
   python mnist_data_selection.py --selector label_based --epochs 10 --top_k 1000

2. 使用特征MLP模型:
   python mnist_data_selection.py --selector feature_mlp --epochs 15

3. 使用CNN特征模型:
   python mnist_data_selection.py --selector cnn_feature --epochs 20

4. 自定义参数:
   python mnist_data_selection.py --selector label_based --epochs 10 --batch_size 128 --top_k 200 --target_digit 9

📊 性能建议:
   - 初学者: 使用 --selector label_based
   - 想要更精细控制: 使用 --selector feature_mlp
   - 追求最高性能: 使用 --selector cnn_feature (需要更多时间)

🎯 预期结果:
   - label_based: 会学习到哪些数字类别对识别数字7最有帮助
   - feature_mlp/cnn_feature: 会学习到更细粒度的样本重要性
   - 最终可视化展示最重要的训练样本
"""
