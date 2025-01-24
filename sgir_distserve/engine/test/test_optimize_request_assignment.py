import time

def check_if_rebalance_available(sequences, now):
    
    if len(sequences)<=4:
    # 简化：假设所有序列都是有效的，并按slack time排序
        return sorted(sequences, key=lambda x: x[1])  # x[1] 是 slack time
    else:
        return []

def optimize_request_assignment(fast_instance_seqs, slow_instance_seqs, fast_instance_id, slow_instance_id):
    # 获取当前时间
    now = time.time()

    # 初始化初始请求ID集合
    init_fast_request_ids_set = set(seq_group['request_id'] for seq_group, _ in fast_instance_seqs)
    init_slow_request_ids_set = set(seq_group['request_id'] for seq_group, _ in slow_instance_seqs)
    
    # 合并请求并按 slack 从大到小排序
    combined_seqs = fast_instance_seqs + slow_instance_seqs
    combined_seqs.sort(key=lambda x: x[1], reverse=True)
    
    # 初始化
    total_requests = len(combined_seqs)
    optimal_fast_seqs = []
    optimal_slow_seqs = []
    max_sum = float('-inf')  # 初始设置为负无穷大

    # 循环尝试不同的分配
    for K in range(1, total_requests):  # 从1开始到 total_requests-1
        # 分配前 K 个给 fast_instance_seqs，其余的给 slow_instance_seqs
        new_fast_seqs = combined_seqs[:K]
        new_slow_seqs = combined_seqs[K:]

        # 检查是否满足 SLO
        new_fast_seqs = check_if_rebalance_available(new_fast_seqs, now)
        new_slow_seqs = check_if_rebalance_available(new_slow_seqs, now)
        
        if not new_fast_seqs or not new_slow_seqs:
            continue

        # 获取新的 min slack 值
        new_min_slack_fast = new_fast_seqs[0][1]
        new_min_slack_slow = new_slow_seqs[0][1]

        # 如果新的 min slack 之和比当前最大值大，更新最优解
        if new_min_slack_fast + new_min_slack_slow > max_sum:
            optimal_fast_seqs = new_fast_seqs
            optimal_slow_seqs = new_slow_seqs
            max_sum = new_min_slack_fast + new_min_slack_slow
    
    # 如果找到了更优的分配方案，更新 fast 和 slow 的请求
    if optimal_fast_seqs and optimal_slow_seqs:
        fast_seqs = optimal_fast_seqs
        slow_seqs = optimal_slow_seqs
    else:
        fast_seqs = fast_instance_seqs
        slow_seqs = slow_instance_seqs
    
    # 准备发送的请求列表
    send_slow2fast_seqs = [
        (seq_group, fast_instance_id) 
        for seq_group, _ in fast_seqs 
        if seq_group['request_id'] in init_slow_request_ids_set
    ]
    send_fast2slow_seqs = [
        (seq_group, slow_instance_id) 
        for seq_group, _ in slow_seqs 
        if seq_group['request_id'] in init_fast_request_ids_set
    ]

    return send_fast2slow_seqs, send_slow2fast_seqs

# 示例请求
fast_instance_seqs = [
    ({'request_id': 1}, 0.07222188307495114), 
    ({'request_id': 2}, 0.8460635471954345), 
    ({'request_id': 3}, 2.9739313128082276), 
    ({'request_id': 4}, 3.0262246847442626)
]
slow_instance_seqs = [
    ({'request_id': 5}, 0.1103459250946044), 
    ({'request_id': 6}, 0.9103459250946044), 
    ({'request_id': 7}, 1.4945747310943602)
]

fast_instance_id = 1
slow_instance_id = 2

# 调用函数
send_fast2slow_seqs, send_slow2fast_seqs = optimize_request_assignment(
    fast_instance_seqs, 
    slow_instance_seqs, 
    fast_instance_id, 
    slow_instance_id
)

print("Sequences to send from fast to slow:", send_fast2slow_seqs)
print("Sequences to send from slow to fast:", send_slow2fast_seqs)
