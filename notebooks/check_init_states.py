#!/usr/bin/env python3
"""
检查 LIBERO 初始状态文件的完整性和有效性
"""

import os
import pickle
import numpy as np
from pathlib import Path
import sys

def check_init_file(file_path):
    """检查单个初始状态文件"""
    results = {
        'file': file_path.name,
        'exists': file_path.exists(),
        'size_kb': 0,
        'num_states': 0,
        'state_keys': [],
        'has_robot_states': False,
        'has_object_states': False,
        'error': None
    }
    
    if not file_path.exists():
        results['error'] = 'File not found'
        return results
    
    try:
        # 获取文件大小
        results['size_kb'] = file_path.stat().st_size / 1024
        
        # 加载 pickle 文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 检查数据结构
        if isinstance(data, list):
            results['num_states'] = len(data)
            if len(data) > 0:
                first_state = data[0]
                if isinstance(first_state, dict):
                    results['state_keys'] = list(first_state.keys())
                    
                    # 检查关键字段
                    results['has_robot_states'] = 'states' in first_state
                    results['has_object_states'] = 'model' in first_state
                    
                    # 详细检查机器人状态
                    if 'states' in first_state:
                        robot_states = first_state['states']
                        if isinstance(robot_states, (list, np.ndarray)):
                            results['robot_state_dim'] = len(robot_states)
                        else:
                            results['robot_state_type'] = type(robot_states).__name__
                    
                    # 检查物体状态
                    if 'model' in first_state:
                        results['has_model_xml'] = True
                else:
                    results['error'] = f'First state is not dict, got {type(first_state)}'
        else:
            results['error'] = f'Data is not list, got {type(data)}'
            
    except Exception as e:
        results['error'] = str(e)
    
    return results

def analyze_task_suite(suite_name='libero_10_eval'):
    """分析整个任务套件的初始状态"""
    base_path = Path('/home/whu/LIBERO_PLUS/libero/libero/init_files') / suite_name
    
    print(f"{'='*80}")
    print(f"检查任务套件: {suite_name}")
    print(f"路径: {base_path}")
    print(f"{'='*80}\n")
    
    if not base_path.exists():
        print(f"❌ 目录不存在: {base_path}")
        return
    
    # 查找所有 .init 和 .pruned_init 文件
    init_files = sorted(base_path.glob('*.init'))
    pruned_files = sorted(base_path.glob('*.pruned_init'))
    
    print(f"📊 文件统计:")
    print(f"  .init 文件: {len(init_files)}")
    print(f"  .pruned_init 文件: {len(pruned_files)}")
    print()
    
    # 检查配对
    init_names = {f.stem for f in init_files}
    pruned_names = {f.stem.replace('.pruned', '') for f in pruned_files}
    
    missing_pruned = init_names - pruned_names
    missing_init = pruned_names - init_names
    
    if missing_pruned:
        print(f"⚠️  缺少 .pruned_init 的任务 ({len(missing_pruned)}):")
        for name in sorted(missing_pruned)[:5]:
            print(f"    - {name}")
        if len(missing_pruned) > 5:
            print(f"    ... 还有 {len(missing_pruned)-5} 个")
        print()
    
    if missing_init:
        print(f"⚠️  缺少 .init 的任务 ({len(missing_init)}):")
        for name in sorted(missing_init)[:5]:
            print(f"    - {name}")
        if len(missing_init) > 5:
            print(f"    ... 还有 {len(missing_init)-5} 个")
        print()
    
    # 详细检查前3个文件
    print(f"🔍 详细检查示例文件:\n")
    
    files_to_check = list(pruned_files[:3]) if pruned_files else list(init_files[:3])
    
    for i, file_path in enumerate(files_to_check, 1):
        results = check_init_file(file_path)
        
        print(f"[{i}] {results['file']}")
        print(f"    大小: {results['size_kb']:.2f} KB")
        print(f"    状态数量: {results['num_states']}")
        
        if results['error']:
            print(f"    ❌ 错误: {results['error']}")
        else:
            print(f"    ✓ 机器人状态: {'是' if results['has_robot_states'] else '否'}")
            print(f"    ✓ 物体模型: {'是' if results['has_object_states'] else '否'}")
            if 'robot_state_dim' in results:
                print(f"    ✓ 状态维度: {results['robot_state_dim']}")
            if results['state_keys']:
                print(f"    ✓ 字段: {', '.join(results['state_keys'][:5])}")
        print()
    
    # 统计分析
    print(f"📈 统计分析:")
    
    all_results = [check_init_file(f) for f in pruned_files[:10]]
    
    valid_files = [r for r in all_results if not r['error']]
    error_files = [r for r in all_results if r['error']]
    
    print(f"  有效文件: {len(valid_files)}/{len(all_results)}")
    print(f"  错误文件: {len(error_files)}/{len(all_results)}")
    
    if valid_files:
        sizes = [r['size_kb'] for r in valid_files]
        state_counts = [r['num_states'] for r in valid_files]
        
        print(f"  平均文件大小: {np.mean(sizes):.2f} KB (范围: {np.min(sizes):.2f} - {np.max(sizes):.2f})")
        print(f"  平均状态数: {np.mean(state_counts):.1f} (范围: {np.min(state_counts)} - {np.max(state_counts)})")
    
    if error_files:
        print(f"\n❌ 错误文件列表:")
        for r in error_files:
            print(f"    - {r['file']}: {r['error']}")
    
    print(f"\n{'='*80}")
    print(f"✅ 检查完成!")
    print(f"{'='*80}")

def test_load_with_libero(suite_name='libero_10_eval', task_idx=0):
    """使用 LIBERO API 测试加载初始状态"""
    print(f"\n{'='*80}")
    print(f"使用 LIBERO API 测试加载")
    print(f"{'='*80}\n")
    
    try:
        from libero.libero import benchmark
        
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite_name]()
        
        print(f"✓ 任务套件: {suite_name}")
        print(f"✓ 任务总数: {task_suite.n_tasks}")
        
        task = task_suite.get_task(task_idx)
        print(f"✓ 测试任务: {task.name}")
        
        init_states = task_suite.get_task_init_states(task_idx)
        print(f"✓ 加载初始状态: {len(init_states)} 个")
        
        # 检查第一个状态
        if len(init_states) > 0:
            state = init_states[0]
            print(f"\n初始状态结构:")
            if isinstance(state, dict):
                for key in state.keys():
                    value = state[key]
                    if isinstance(value, (list, np.ndarray)):
                        print(f"  {key}: {type(value).__name__} shape={np.array(value).shape}")
                    else:
                        print(f"  {key}: {type(value).__name__}")
        
        print(f"\n✅ LIBERO API 加载成功!")
        return True
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    suite = sys.argv[1] if len(sys.argv) > 1 else 'libero_10_eval'
    
    # 分析文件
    analyze_task_suite(suite)
    
    # 测试 LIBERO API
    test_load_with_libero(suite, task_idx=0)
