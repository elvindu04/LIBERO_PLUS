#!/usr/bin/env python3
"""
可视化检查 LIBERO 初始状态
"""

import sys
sys.path.insert(0, '/home/whu/LIBERO_PLUS')
sys.path.insert(0, '/home/whu/SimpleVLA-RL')

from libero.libero import benchmark
from verl.utils.libero_utils import get_libero_env
import numpy as np
import matplotlib.pyplot as plt

def visualize_init_states(suite_name='libero_10_eval', task_idx=0, num_samples=5):
    """可视化任务的初始状态"""
    print(f"加载任务套件: {suite_name}")
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_idx)
    
    print(f"任务: {task.name}")
    print(f"描述: {task.language}")
    
    # 加载初始状态
    init_states = task_suite.get_task_init_states(task_idx)
    print(f"初始状态数量: {len(init_states)}")
    
    # 创建环境
    env, task_description = get_libero_env(task, 'openvla', resolution=256)
    
    print(f"\n采样 {num_samples} 个初始状态:")
    images = []
    
    for i in range(min(num_samples, len(init_states))):
        print(f"  [{i+1}] 设置初始状态 {i}...")
        
        obs = env.reset()
        env.set_init_state(init_states[i])
        
        # 获取观察图像
        img = obs.get('agentview_image', obs.get('image', None))
        if img is not None:
            images.append(img)
            
            # 检查状态信息
            if 'robot0_eef_pos' in obs:
                eef_pos = obs['robot0_eef_pos']
                print(f"      末端位置: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]")
    
    env.close()
    
    # 绘制图像
    if images:
        fig, axes = plt.subplots(1, len(images), figsize=(4*len(images), 4))
        if len(images) == 1:
            axes = [axes]
        
        for i, (ax, img) in enumerate(zip(axes, images)):
            ax.imshow(img)
            ax.set_title(f'Init State {i}')
            ax.axis('off')
        
        plt.tight_layout()
        output_path = f'/home/whu/LIBERO_PLUS/notebooks/init_states_vis_{suite_name}_{task_idx}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 保存可视化: {output_path}")
        plt.close()
    
    return True

def test_task_execution(suite_name='libero_10_eval', task_idx=0, num_steps=50):
    """测试任务执行和成功检测"""
    print(f"\n{'='*80}")
    print(f"测试任务执行")
    print(f"{'='*80}\n")
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_idx)
    init_states = task_suite.get_task_init_states(task_idx)
    
    env, task_description = get_libero_env(task, 'openvla', resolution=256)
    
    print(f"任务: {task.name}")
    print(f"测试 {num_steps} 步随机动作...\n")
    
    # 测试3个不同初始状态
    for trial in range(3):
        obs = env.reset()
        env.set_init_state(init_states[trial])
        
        done = False
        for step in range(num_steps):
            action = np.random.randn(7) * 0.05  # 小幅度随机动作
            obs, reward, done, info = env.step(action)
            
            if done:
                print(f"  Trial {trial}: ✓ 在第 {step} 步完成! reward={reward:.3f}")
                break
        
        if not done:
            print(f"  Trial {trial}: ✗ {num_steps} 步未完成")
    
    env.close()
    print()

if __name__ == '__main__':
    suite = sys.argv[1] if len(sys.argv) > 1 else 'libero_10_eval'
    task_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    # 可视化初始状态
    visualize_init_states(suite, task_idx, num_samples=5)
    
    # 测试执行
    test_task_execution(suite, task_idx, num_steps=100)
