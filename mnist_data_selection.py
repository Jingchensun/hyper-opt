"""
MNIST数据选择示例：
根据数字6和7的验证损失，从0-9的训练样本中选择最相关的样本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from model import DataSelectionHyperOptModel
from hyper_opt import NeumannHyperOptimizer, FixedPointHyperOptimizer
from network import SimpleModel


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
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        
        print(f"训练样本数量: {len(train_indices)}")
        print(f"验证样本数量 (数字{self.target_digits}): {len(val_indices)}")
        
        return train_loader, val_loader, train_indices
    
    def train_step(self, train_data, train_targets, train_batch_indices):
        """执行一步训练"""
        
        # 定义训练损失函数
        def train_loss_func():
            # 重新计算训练损失（用于随机模式）
            train_loss, train_logit = self.model.train_loss(
                train_data, train_targets, train_batch_indices
            )
            return train_loss, train_logit
        
        # 计算验证损失
        val_loss = self._compute_validation_loss()
        
        # 超参数优化步骤
        self.hyper_optimizer.step(train_loss_func, val_loss)
        
        # 模型参数优化步骤
        self.model_optimizer.zero_grad()
        train_loss, _ = train_loss_func()
        train_loss.backward()
        self.model_optimizer.step()
        
        return train_loss.item(), val_loss.item()
    
    def _compute_validation_loss(self):
        """计算验证损失"""
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for val_data, val_targets in self.val_loader:
                val_data, val_targets = val_data.to(self.device), val_targets.to(self.device)
                val_loss = self.model.validation_loss(val_data, val_targets)
                total_loss += val_loss.item() * val_data.size(0)
                total_samples += val_data.size(0)
        
        return torch.tensor(total_loss / total_samples, requires_grad=True)
    
    def train(self, num_epochs=50):
        """训练数据选择模型"""
        train_losses = []
        val_losses = []
        
        print("开始训练数据选择模型...")
        
        for epoch in range(num_epochs):
            epoch_train_loss = 0
            epoch_val_loss = 0
            num_batches = 0
            
            for batch_idx, (train_data, train_targets) in enumerate(self.train_loader):
                train_data, train_targets = train_data.to(self.device), train_targets.to(self.device)
                
                # 计算当前batch在整个训练集中的索引
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.train_indices))
                train_batch_indices = torch.arange(start_idx, end_idx)
                
                # 执行训练步骤
                train_loss, val_loss = self.train_step(
                    train_data, train_targets, train_batch_indices
                )
                
                epoch_train_loss += train_loss
                epoch_val_loss += val_loss
                num_batches += 1
            
            avg_train_loss = epoch_train_loss / num_batches
            avg_val_loss = epoch_val_loss / num_batches
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        return train_losses, val_losses
    
    def analyze_sample_importance(self, top_k=100):
        """分析样本重要性并返回最重要的样本"""
        
        # 获取所有样本的权重
        weights = self.model.get_sample_weights().detach().cpu().numpy()
        
        # 找到权重最高的样本
        top_indices = np.argsort(weights)[-top_k:][::-1]
        top_weights = weights[top_indices]
        
        print(f"\n前{top_k}个最重要样本的权重:")
        print(f"最高权重: {top_weights[0]:.4f}")
        print(f"最低权重: {weights.min():.4f}")
        print(f"平均权重: {weights.mean():.4f}")
        print(f"权重标准差: {weights.std():.4f}")
        
        return top_indices, top_weights
    
    def visualize_results(self, top_indices, num_samples=20):
        """可视化最重要的样本"""
        
        # 加载原始数据集用于可视化
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST('data', train=True, transform=transform)
        
        fig, axes = plt.subplots(4, 5, figsize=(12, 10))
        fig.suptitle('最重要的训练样本 (基于6&7验证损失)', fontsize=16)
        
        for i in range(min(num_samples, len(top_indices))):
            row = i // 5
            col = i % 5
            
            # 获取真实的数据集索引
            real_idx = self.train_indices[top_indices[i]]
            image, label = dataset[real_idx]
            
            axes[row, col].imshow(image.squeeze(), cmap='gray')
            axes[row, col].set_title(f'数字: {label}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """主函数：演示MNIST数据选择"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据选择器
    selector = MNISTDataSelector(target_digits=[6, 7], batch_size=64, device=device)
    
    # 训练模型
    train_losses, val_losses = selector.train(num_epochs=30)
    
    # 分析样本重要性
    top_indices, top_weights = selector.analyze_sample_importance(top_k=100)
    
    # 可视化结果
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
