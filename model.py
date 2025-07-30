from network import UNet
import torch
import torch.nn as nn
from torch.nn.modules.container import ParameterList

class BaseHyperOptModel(nn.Module):
    
    # TODO: how to implement this like decorator pattern. If we have multiple approaches 
    def __init__(self, network, criterion) -> None:
        super().__init__()
        
        self.network = network
        self.criterion = criterion
    
    @property
    def parameters(self):
        return list(self.network.parameters())
    
    @property
    def hyper_parameters(self):
        raise NotImplementedError
    
    def train_loss(self, x, y):
        
        x = self.data_augment(x, y)
        
        logit = self.network(x)
        loss = self.criterion(logit, y)
        regularizer = self.regularizer()
        
        return loss + regularizer, logit
    
    
    
    def validation_loss(self, x, y):
        
        logit = self.network(x)
        loss = self.criterion(logit, y)
        return loss
    
    def data_augment(self, x, y):
        """
        Overwrite this to perform data_augmentation task
        """
        return x
    
    def regularizer(self):
        """
        Overwrite this for customizing regularizer
        """
        return 0.
        
class L2HyperOptModel(BaseHyperOptModel):
    
    def __init__(self, network, criterion) -> None:
        super().__init__(network, criterion)
        self.l2 = nn.Parameter(torch.tensor([1e-3]))
        
    @property
    def hyper_parameters(self):
        return [self.l2]
    
    def regularizer(self):
        
        ret = 0.
        for param in self.parameters:
            ret += torch.norm(param)
        return ret * (self.l2)
    

class AllL2HyperOptModel(BaseHyperOptModel):
    
    def __init__(self, network, criterion) -> None:
        super().__init__(network, criterion)
        
        weight_decay = []
        for param in self.network.parameters():
            l2 = nn.Parameter(torch.ones_like(param)*(-5))
            weight_decay.append(l2)
        
        self.weight_decay = ParameterList(weight_decay)
    
    @property
    def hyper_parameters(self):
        return list(self.weight_decay.parameters())
    
    def regularizer(self):
        ret = 0.
        for weight_decay, param in zip(self.weight_decay, self.parameters):
            l2 = weight_decay.exp()
            ret += torch.sum((l2 * param)**2)
        return ret
    

class UNetAugmentHyperOptModel(BaseHyperOptModel):
    
    def __init__(self, network, criterion,
                 in_channels,
                 num_classes,
                 depth=2,
                 wf=3,
                 padding=True,
                 batch_norm=False,
                 do_noise_channel=True,
                 up_mode='upconv',
                 use_indentity_residual=True) -> None:
        super().__init__(network, criterion)
        
        self.augment_net = UNet(in_channels, num_classes,
                                depth,
                                wf=wf,
                                padding=padding,
                                batch_norm=batch_norm,
                                do_noise_channel=do_noise_channel,
                                use_identity_residual=use_indentity_residual,
                                up_mode=up_mode)
        
    
    @property
    def hyper_parameters(self):
        return list(self.augment_net.parameters())
    
    def data_augment(self, x, y):
        return self.augment_net(x, y)
            
    
class ReweightHyperOptModel(BaseHyperOptModel):
    
    def __init__(self, network, criterion) -> None:
        super().__init__(network, criterion)


class DataSelectionHyperOptModel(BaseHyperOptModel):
    """
    数据选择模型：为每个训练样本分配可学习的重要性权重
    用于根据验证损失自动选择最相关的训练样本
    """
    
    def __init__(self, network, criterion, num_train_samples, initial_weight=1.0) -> None:
        super().__init__(network, criterion)
        
        # 为每个训练样本创建可学习的权重参数
        # 使用log空间来确保权重为正数
        self.sample_weights = nn.Parameter(
            torch.ones(num_train_samples) * torch.log(torch.tensor(initial_weight))
        )
        
        self.num_train_samples = num_train_samples
        
    @property
    def hyper_parameters(self):
        return [self.sample_weights]
    
    def get_sample_weights(self):
        """获取当前的样本权重（转换为正数）"""
        return torch.exp(self.sample_weights)
    
    def train_loss(self, x, y, sample_indices=None):
        """
        计算加权的训练损失
        
        Args:
            x: 输入数据
            y: 标签
            sample_indices: 当前batch中样本在整个训练集中的索引
        """
        if sample_indices is None:
            # 如果没有提供索引，假设是按顺序的
            batch_size = x.shape[0]
            sample_indices = torch.arange(batch_size)
        
        # 数据增强
        x = self.data_augment(x, y)
        
        # 前向传播
        logit = self.network(x)
        
        # 计算每个样本的损失（不进行reduction）
        losses = self.criterion(logit, y)
        
        # 获取当前batch样本的权重
        current_weights = self.get_sample_weights()[sample_indices]
        
        # 加权损失
        weighted_loss = torch.mean(losses * current_weights)
        
        # 添加正则化项
        regularizer = self.regularizer()
        
        return weighted_loss + regularizer, logit
    
    def validation_loss(self, x, y):
        """
        计算验证损失（返回标量）
        """
        logit = self.network(x)
        # 计算每个样本的损失，然后取平均
        losses = self.criterion(logit, y)
        return torch.mean(losses)


class LabelBasedDataSelectionModel(BaseHyperOptModel):
    """
    基于标签的简化数据选择模型：为每个数字类别(0-9)分配权重
    参数量：从60,000个减少到10个
    """
    
    def __init__(self, network, criterion, num_classes=10, initial_weight=1.0) -> None:
        super().__init__(network, criterion)
        
        # 为每个类别创建可学习权重（在log空间确保正数）
        self.class_weights = nn.Parameter(
            torch.ones(num_classes) * torch.log(torch.tensor(initial_weight))
        )
        
        self.num_classes = num_classes
        
    @property
    def hyper_parameters(self):
        return [self.class_weights]
    
    def get_class_weights(self):
        """获取当前的类别权重（转换为正数）"""
        return torch.exp(self.class_weights)
    
    def get_sample_weights_by_labels(self, labels):
        """根据标签获取样本权重"""
        weights = self.get_class_weights()
        return weights[labels]
    
    def train_loss(self, x, y, sample_indices=None):
        """计算基于类别权重的训练损失"""
        # 数据增强
        x = self.data_augment(x, y)
        
        # 前向传播
        logit = self.network(x)
        
        # 计算每个样本的损失
        losses = self.criterion(logit, y)
        
        # 根据标签获取权重
        current_weights = self.get_sample_weights_by_labels(y)
        
        # 加权损失
        weighted_loss = torch.mean(losses * current_weights)
        
        # 添加正则化项
        regularizer = self.regularizer()
        
        return weighted_loss + regularizer, logit
    
    def validation_loss(self, x, y):
        """计算验证损失"""
        logit = self.network(x)
        losses = self.criterion(logit, y)
        return torch.mean(losses)


class FeatureMlpDataSelectionModel(BaseHyperOptModel):
    """
    基于特征MLP的数据选择模型：使用简单统计特征预测样本权重
    参数量：约几百个参数
    """
    
    def __init__(self, network, criterion, feature_dim=16, hidden_dim=32) -> None:
        super().__init__(network, criterion)
        
        # 权重预测网络
        self.weight_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 确保权重在0-1之间
        )
        
        self.feature_dim = feature_dim
        
    @property
    def hyper_parameters(self):
        return list(self.weight_network.parameters())
    
    def extract_features(self, x):
        """
        提取简单的统计特征
        Args:
            x: 输入图像 [batch_size, channels, height, width]
        Returns:
            features: [batch_size, feature_dim]
        """
        batch_size = x.shape[0]
        features = []
        
        for i in range(batch_size):
            img = x[i].squeeze()  # [H, W]
            
            # 基本统计特征
            mean_val = torch.mean(img)
            std_val = torch.std(img)
            min_val = torch.min(img)
            max_val = torch.max(img)
            
            # 分区域统计（将图像分为4个区域）
            h, w = img.shape
            h_mid, w_mid = h // 2, w // 2
            
            # 四个区域的均值
            top_left = torch.mean(img[:h_mid, :w_mid])
            top_right = torch.mean(img[:h_mid, w_mid:])
            bottom_left = torch.mean(img[h_mid:, :w_mid])
            bottom_right = torch.mean(img[h_mid:, w_mid:])
            
            # 边缘检测特征（简单差分）
            horizontal_edges = torch.mean(torch.abs(img[1:] - img[:-1]))
            vertical_edges = torch.mean(torch.abs(img[:, 1:] - img[:, :-1]))
            
            # 中心区域 vs 边缘区域
            center_region = img[h//4:3*h//4, w//4:3*w//4]
            center_mean = torch.mean(center_region)
            
            # 对称性特征
            left_half = img[:, :w//2]
            right_half = torch.flip(img[:, w//2:], dims=[1])
            if left_half.shape[1] == right_half.shape[1]:
                symmetry = -torch.mean(torch.abs(left_half - right_half))
            else:
                symmetry = torch.tensor(0.0, device=x.device)
            
            # 组合特征
            feature_vector = torch.stack([
                mean_val, std_val, min_val, max_val,
                top_left, top_right, bottom_left, bottom_right,
                horizontal_edges, vertical_edges, center_mean, symmetry,
                # 添加一些交互特征
                mean_val * std_val,
                max_val - min_val,
                center_mean - mean_val,
                torch.tensor(0.0, device=x.device)  # padding to feature_dim=16
            ])
            
            features.append(feature_vector)
        
        return torch.stack(features)
    
    def get_sample_weights(self, x):
        """根据输入特征计算样本权重"""
        features = self.extract_features(x)
        weights = self.weight_network(features).squeeze(-1)
        return weights
    
    def train_loss(self, x, y, sample_indices=None):
        """计算基于特征预测权重的训练损失"""
        # 数据增强
        x_aug = self.data_augment(x, y)
        
        # 前向传播
        logit = self.network(x_aug)
        
        # 计算每个样本的损失
        losses = self.criterion(logit, y)
        
        # 使用原始输入计算权重（不用增强的数据）
        current_weights = self.get_sample_weights(x)
        
        # 加权损失
        weighted_loss = torch.mean(losses * current_weights)
        
        # 添加正则化项
        regularizer = self.regularizer()
        
        return weighted_loss + regularizer, logit
    
    def validation_loss(self, x, y):
        """计算验证损失"""
        logit = self.network(x)
        losses = self.criterion(logit, y)
        return torch.mean(losses)


class CnnFeatureDataSelectionModel(BaseHyperOptModel):
    """
    基于CNN特征的数据选择模型：使用网络中间特征预测样本权重
    参数量：中等，复用主网络的特征提取能力
    """
    
    def __init__(self, network, criterion, hidden_dim=64) -> None:
        super().__init__(network, criterion)
        
        # 假设主网络有特征提取能力，我们添加一个小的权重预测头
        # 这里假设可以从网络中获取特征，具体实现取决于network的结构
        
        # 权重预测网络（输入维度需要根据实际的特征维度调整）
        self.weight_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 特征提取层（从主网络复制前几层或者使用共享特征）
        # 这里创建一个简单的特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
    @property
    def hyper_parameters(self):
        return list(self.weight_head.parameters()) + list(self.feature_extractor.parameters())
    
    def get_sample_weights(self, x):
        """基于CNN特征计算样本权重"""
        features = self.feature_extractor(x)
        weights = self.weight_head(features).squeeze(-1)
        return weights
    
    def train_loss(self, x, y, sample_indices=None):
        """计算基于CNN特征权重的训练损失"""
        # 数据增强
        x_aug = self.data_augment(x, y)
        
        # 前向传播
        logit = self.network(x_aug)
        
        # 计算每个样本的损失
        losses = self.criterion(logit, y)
        
        # 使用原始输入计算权重
        current_weights = self.get_sample_weights(x)
        
        # 加权损失
        weighted_loss = torch.mean(losses * current_weights)
        
        # 添加正则化项
        regularizer = self.regularizer()
        
        return weighted_loss + regularizer, logit
    
    def validation_loss(self, x, y):
        """计算验证损失"""
        logit = self.network(x)
        losses = self.criterion(logit, y)
        return torch.mean(losses)