#!/usr/bin/env python3
"""
快速测试MNIST数据选择 - 优化版本
"""

import subprocess
import sys

def run_quick_test():
    """运行快速测试，大约1-2分钟完成"""
    
    print("🚀 启动快速测试...")
    print("⏱️  预计运行时间: 1-2分钟")
    print("=" * 50)
    
    # 最优化的参数设置
    cmd = [
        sys.executable, "mnist_data_selection.py",
        "--selector", "label_based",      # 最快的模型（只有10个参数）
        "--epochs", "2",                  # 很少的轮数  
        "--batch_size", "512",            # 更大的batch减少迭代次数
        "--fast_mode",                    # 启用快速模式
        "--val_subset_size", "500",       # 大幅减少验证集大小
        "--hyper_opt_freq", "20",         # 减少超参数优化频率
        "--top_k", "50",                  # 减少分析的样本数
        "--vis_samples", "12"             # 减少可视化样本数
    ]
    
    print("🔧 使用的优化参数:")
    print("   - 模型类型: label_based (仅10个超参数)")
    print("   - 训练轮数: 2") 
    print("   - 批次大小: 512 (减少batch数量)")
    print("   - 快速模式: 启用")
    print("   - 验证集子集: 500 (从6000减少到500)")
    print("   - 超参数优化频率: 每20个batch一次")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ 快速测试完成!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 测试失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
        return False
        
    return True

def run_ultra_fast_test():
    """超快速测试，约30秒完成"""
    
    print("⚡ 启动超快速测试...")
    print("⏱️  预计运行时间: 30秒")
    print("=" * 50)
    
    cmd = [
        sys.executable, "mnist_data_selection.py", 
        "--selector", "label_based",
        "--epochs", "1",                  # 只训练1轮
        "--batch_size", "1024",           # 很大的batch
        "--fast_mode",
        "--val_subset_size", "200",       # 极小的验证集
        "--hyper_opt_freq", "50",         # 很少的超参数优化
        "--top_k", "20",
        "--vis_samples", "8"
    ]
    
    print("🔧 使用的超快速参数:")
    print("   - 训练轮数: 1 (最少)")
    print("   - 批次大小: 1024 (最大)")  
    print("   - 验证集子集: 200 (极小)")
    print("   - 超参数优化频率: 每50个batch一次")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n⚡ 超快速测试完成!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 测试失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
        return False
        
    return True

if __name__ == "__main__":
    print("🎯 MNIST数据选择 - 快速测试工具")
    print()
    print("选择测试模式:")
    print("1. 快速测试 (1-2分钟，推荐)")
    print("2. 超快速测试 (30秒，演示用)")
    print("3. 退出")
    
    while True:
        choice = input("\n请选择 (1/2/3): ").strip()
        
        if choice == "1":
            run_quick_test()
            break
        elif choice == "2": 
            run_ultra_fast_test()
            break
        elif choice == "3":
            print("👋 退出")
            break
        else:
            print("❌ 无效选择，请输入1、2或3") 