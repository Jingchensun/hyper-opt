#!/usr/bin/env python3
"""
MNIST数据选择结果可视化分析工具
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from torchvision import datasets, transforms
from collections import Counter
import seaborn as sns

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
try:
    sns.set_style("whitegrid")
    plt.style.use('seaborn-v0_8')
except:
    # 如果seaborn-v0_8不可用，使用默认样式
    try:
        plt.style.use('seaborn')
    except:
        pass  # 使用默认matplotlib样式

class MNISTResultVisualizer:
    def __init__(self, selector, train_losses, val_losses, all_weights, top_indices):
        self.selector = selector
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.all_weights = all_weights
        self.top_indices = top_indices
        
        # 加载数据集用于可视化
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = datasets.MNIST('data', train=True, transform=self.transform)
        
    def create_comprehensive_analysis(self):
        """创建综合分析图"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 训练损失曲线
        plt.subplot(3, 4, 1)
        self.plot_training_curves()
        
        # 2. 权重分布直方图
        plt.subplot(3, 4, 2)
        self.plot_weight_distribution()
        
        # 3. 各数字类别权重对比
        plt.subplot(3, 4, 3)
        self.plot_class_weights()
        
        # 4. Top样本数字分布
        plt.subplot(3, 4, 4)
        self.plot_top_samples_distribution()
        
        # 5-8. 最重要样本展示 (2x2)
        self.plot_top_samples_grid(start_subplot=5)
        
        # 9. 权重vs数字类别的箱线图
        plt.subplot(3, 4, 9)
        self.plot_weight_by_digit_boxplot()
        
        # 10. 低权重样本展示
        plt.subplot(3, 4, 10)
        self.plot_low_weight_samples()
        
        # 11. 权重变化热力图（如果是label_based）
        plt.subplot(3, 4, 11)
        self.plot_class_weight_heatmap()
        
        # 12. 统计总结
        plt.subplot(3, 4, 12)
        self.plot_statistics_summary()
        
        plt.suptitle('MNIST Data Selection Analysis Results', fontsize=20, y=0.98)
        plt.tight_layout()
        plt.savefig('mnist_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_training_curves(self):
        """绘制训练曲线"""
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
        plt.plot(epochs, self.val_losses, 'r-s', label='Validation Loss (Digit 7)', linewidth=2, markersize=4)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加趋势标注
        if len(self.train_losses) > 1:
            train_trend = "↓" if self.train_losses[-1] < self.train_losses[0] else "↑"
            val_trend = "↓" if self.val_losses[-1] < self.val_losses[0] else "↑"
            plt.text(0.05, 0.95, f'Train {train_trend} Val {val_trend}', 
                    transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    def plot_weight_distribution(self):
        """绘制权重分布"""
        plt.hist(self.all_weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.all_weights.mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.all_weights.mean():.3f}')
        plt.axvline(self.all_weights.mean() + self.all_weights.std(), color='orange', 
                   linestyle='--', label=f'Mean+Std: {self.all_weights.mean() + self.all_weights.std():.3f}')
        
        plt.xlabel('Sample Weight')
        plt.ylabel('Frequency')
        plt.title('Sample Weight Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    def plot_class_weights(self):
        """绘制各数字类别的权重"""
        if self.selector.model_type == 'label_based':
            class_weights = self.selector.model.get_class_weights().detach().cpu().numpy()
            digits = list(range(10))
            
            bars = plt.bar(digits, class_weights, color='lightcoral', edgecolor='black')
            plt.xlabel('Digit Class')
            plt.ylabel('Class Weight')
            plt.title('Weight by Digit Class')
            plt.xticks(digits)
            
            # 高亮数字7
            bars[7].set_color('gold')
            bars[7].set_edgecolor('red')
            bars[7].set_linewidth(2)
            
            # 添加数值标签
            for i, v in enumerate(class_weights):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Class weights not available\nfor this model type', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Class Weights (N/A)')
    
    def plot_top_samples_distribution(self):
        """绘制Top样本的数字分布"""
        top_labels = []
        for idx in self.top_indices[:100]:  # 取前100个
            real_idx = self.selector.train_indices[idx]
            _, label = self.dataset[real_idx]
            top_labels.append(label)
        
        digit_counts = Counter(top_labels)
        digits = list(range(10))
        counts = [digit_counts.get(d, 0) for d in digits]
        
        bars = plt.bar(digits, counts, color='lightgreen', edgecolor='black')
        bars[7].set_color('gold')  # 高亮数字7
        
        plt.xlabel('Digit')
        plt.ylabel('Count in Top 100')
        plt.title('Top Samples Distribution')
        plt.xticks(digits)
        
        # 添加百分比标签
        total = sum(counts)
        for i, v in enumerate(counts):
            if v > 0:
                plt.text(i, v + 0.5, f'{v}\n({v/total*100:.1f}%)', 
                        ha='center', va='bottom', fontsize=8)
        plt.grid(True, alpha=0.3)
    
    def plot_top_samples_grid(self, start_subplot=5):
        """绘制最重要样本的2x2网格"""
        for i in range(4):
            plt.subplot(3, 4, start_subplot + i)
            
            if i < len(self.top_indices):
                real_idx = self.selector.train_indices[self.top_indices[i]]
                image, label = self.dataset[real_idx]
                weight = self.all_weights[self.top_indices[i]]
                
                plt.imshow(image.squeeze(), cmap='gray')
                plt.title(f'Rank {i+1}: Digit {label}\nWeight: {weight:.4f}', fontsize=10)
                plt.axis('off')
            else:
                plt.axis('off')
    
    def plot_weight_by_digit_boxplot(self):
        """绘制各数字权重的箱线图"""
        digit_weights = {i: [] for i in range(10)}
        
        for idx, weight in enumerate(self.all_weights):
            real_idx = self.selector.train_indices[idx]
            _, label = self.dataset[real_idx]
            digit_weights[label].append(weight)
        
        data = [digit_weights[i] for i in range(10)]
        labels = [f'{i}' for i in range(10)]
        
        box_plot = plt.boxplot(data, labels=labels, patch_artist=True)
        
        # 设置颜色
        colors = ['lightblue'] * 10
        colors[7] = 'gold'  # 高亮数字7
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.xlabel('Digit')
        plt.ylabel('Weight')
        plt.title('Weight Distribution by Digit')
        plt.grid(True, alpha=0.3)
    
    def plot_low_weight_samples(self):
        """展示低权重样本"""
        low_indices = np.argsort(self.all_weights)[:4]  # 最低的4个
        
        if len(low_indices) > 0:
            real_idx = self.selector.train_indices[low_indices[0]]
            image, label = self.dataset[real_idx]
            weight = self.all_weights[low_indices[0]]
            
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f'Lowest Weight Sample\nDigit {label}, Weight: {weight:.4f}', fontsize=10)
            plt.axis('off')
        else:
            plt.text(0.5, 0.5, 'No low weight\nsamples found', 
                    ha='center', va='center', transform=plt.gca().transAxes)
    
    def plot_class_weight_heatmap(self):
        """绘制类别权重热力图"""
        if self.selector.model_type == 'label_based':
            class_weights = self.selector.model.get_class_weights().detach().cpu().numpy()
            
            # 创建热力图数据 (1x10)
            heatmap_data = class_weights.reshape(1, -1)
            
            im = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            
            plt.yticks([0], ['Weight'])
            plt.xticks(range(10), [f'Digit {i}' for i in range(10)])
            plt.title('Class Weight Heatmap')
            
            # 添加数值标注
            for i in range(10):
                plt.text(i, 0, f'{class_weights[i]:.3f}', 
                        ha='center', va='center', fontsize=8)
        else:
            plt.text(0.5, 0.5, 'Heatmap not available\nfor this model type', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Weight Heatmap (N/A)')
    
    def plot_statistics_summary(self):
        """绘制统计总结"""
        plt.axis('off')
        
        # 计算统计信息
        stats_text = f"""
        📊 Training Summary
        
        Model Type: {self.selector.model_type}
        Total Samples: {len(self.all_weights):,}
        
        📈 Weight Statistics
        Mean: {self.all_weights.mean():.4f}
        Std: {self.all_weights.std():.4f}
        Min: {self.all_weights.min():.4f}
        Max: {self.all_weights.max():.4f}
        
        🎯 Performance
        Final Train Loss: {self.train_losses[-1]:.4f}
        Final Val Loss: {self.val_losses[-1]:.4f}
        
        🏆 Top Sample Analysis
        Unique weights in top 100: {len(set(self.all_weights[self.top_indices[:100]]))}
        """
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    def create_detailed_sample_analysis(self):
        """创建详细的样本分析图"""
        fig, axes = plt.subplots(4, 10, figsize=(20, 8))
        fig.suptitle('Top 40 Most Important Samples for Digit 7 Recognition', fontsize=16)
        
        for i in range(40):
            row = i // 10
            col = i % 10
            
            if i < len(self.top_indices):
                real_idx = self.selector.train_indices[self.top_indices[i]]
                image, label = self.dataset[real_idx]
                weight = self.all_weights[self.top_indices[i]]
                
                axes[row, col].imshow(image.squeeze(), cmap='gray')
                axes[row, col].set_title(f'#{i+1}\nDigit: {label}\nW: {weight:.3f}', fontsize=8)
                axes[row, col].axis('off')
                
                # 如果是数字7，用红色边框
                if label == 7:
                    for spine in axes[row, col].spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(2)
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('mnist_top_samples_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_result_insights(self):
        """分析结果洞察"""
        print("\n" + "="*60)
        print("🔍 MNIST数据选择结果分析")
        print("="*60)
        
        # 分析权重分布
        unique_weights = len(set(self.all_weights))
        print(f"📊 权重统计:")
        print(f"   - 独特权重数量: {unique_weights:,}")
        print(f"   - 总样本数量: {len(self.all_weights):,}")
        print(f"   - 权重分布范围: [{self.all_weights.min():.6f}, {self.all_weights.max():.6f}]")
        
        # 分析Top样本
        top_labels = []
        for idx in self.top_indices[:100]:
            real_idx = self.selector.train_indices[idx]
            _, label = self.dataset[real_idx]
            top_labels.append(label)
        
        digit_counts = Counter(top_labels)
        print(f"\n🏆 Top 100样本分析:")
        for digit in sorted(digit_counts.keys()):
            percentage = digit_counts[digit] / 100 * 100
            print(f"   - 数字 {digit}: {digit_counts[digit]} 个 ({percentage:.1f}%)")
        
        # 分析模型行为
        print(f"\n🧠 模型行为分析:")
        if self.selector.model_type == 'label_based':
            class_weights = self.selector.model.get_class_weights().detach().cpu().numpy()
            highest_weight_digit = np.argmax(class_weights)
            print(f"   - 最高权重类别: 数字 {highest_weight_digit} (权重: {class_weights[highest_weight_digit]:.6f})")
            print(f"   - 数字7的权重: {class_weights[7]:.6f}")
            print(f"   - 权重比值 (数字7/平均): {class_weights[7]/class_weights.mean():.2f}x")
            
            # 解释为什么所有Top样本权重相同
            if unique_weights <= 10:
                print(f"\n💡 结果解释:")
                print(f"   - 使用label_based模型，同一数字的所有样本权重相同")
                print(f"   - 数字7获得最高权重 {class_weights[7]:.6f}")
                print(f"   - 因此所有数字7样本都是'最重要'的样本")
                print(f"   - 这解释了为什么Top样本都是数字7且权重相同")


def visualize_mnist_results(selector, train_losses, val_losses, all_weights, top_indices):
    """主函数：可视化MNIST结果"""
    
    print("🎨 正在生成详细的可视化分析...")
    
    visualizer = MNISTResultVisualizer(selector, train_losses, val_losses, all_weights, top_indices)
    
    # 创建综合分析图
    print("📊 生成综合分析图...")
    visualizer.create_comprehensive_analysis()
    
    # 创建详细样本分析
    print("🔍 生成详细样本分析...")
    visualizer.create_detailed_sample_analysis()
    
    # 打印分析洞察
    visualizer.analyze_result_insights()
    
    print("\n✅ 可视化分析完成!")
    print("📁 已保存图片文件:")
    print("   - mnist_analysis_comprehensive.png (综合分析)")
    print("   - mnist_top_samples_detailed.png (详细样本分析)")


if __name__ == "__main__":
    print("这是一个可视化工具模块，请从主程序调用 visualize_mnist_results() 函数") 