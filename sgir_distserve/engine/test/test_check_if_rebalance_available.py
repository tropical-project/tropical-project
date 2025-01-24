import unittest
from typing import List, Tuple

# 假设的 DistserveSequenceGroup 替换为字典
class MockDistserveSequenceGroup:
    def __init__(self, request_id, decode_slack):
        self.request_id = request_id
        self.decode_slack = decode_slack

# 模拟的 pd_profiler
class MockPdProfiler:
    def get_decode_seq_prediction(self, seq_list):
        # 这里返回一个固定的预测值，实际中应该是根据 seq_list 计算得出
        return 0.5

    def get_deocode_slack_on_pipeline(self, seq_group, now, tpot_predict):
        # 这里返回 seq_group 的 decode_slack 值，实际中应该是根据 seq_group 和 now 计算得出
        return seq_group.decode_slack

# 测试 check_if_rebalance_available 函数
class TestCheckIfRebalanceAvailable(unittest.TestCase):
    def test_check_if_rebalance_available(self):
        # 创建模拟的序列组和现在的时间
        seqs = [
            (MockDistserveSequenceGroup(request_id=1, decode_slack=0.1), 0.1),
            (MockDistserveSequenceGroup(request_id=2, decode_slack=0.2), 0.2),
            (MockDistserveSequenceGroup(request_id=3, decode_slack=-0.3), -0.3),  # 这个应该会导致函数返回 []
        ]
        now = 1234567890

        # 创建模拟的 pd_profiler 实例
        pd_profiler = MockPdProfiler()

        # 调用 check_if_rebalance_available 函数
        result = check_if_rebalance_available(seqs, now, pd_profiler)

        # 检查结果是否符合预期
        self.assertEqual(len(result), 2)  # 应该排除 decode_slack < -0.2 的序列
        self.assertEqual([result[0][1], result[1][1]], [0.1, 0.2])  # 检查剩余序列的 slack 是否正确

# 定义 check_if_rebalance_available 函数，使用模拟的 pd_profiler
def check_if_rebalance_available(seqs_tuple_list: List[Tuple[dict, float]], now, pd_profiler:MockPdProfiler):
    slack_list = []
    seq_list = [seq[0] for seq in seqs_tuple_list]
    tpot_predict = pd_profiler.get_decode_seq_prediction(seq_list)
    
    for seq_group, _ in seqs_tuple_list:
        decode_slack = pd_profiler.get_deocode_slack_on_pipeline(seq_group, now, tpot_predict)
        slack_list.append((seq_group, decode_slack))
        
        if decode_slack < -0.2:
            return []
            
    sorted_slack_list = sorted(slack_list, key=lambda x: x[1], reverse=False)
    return sorted_slack_list

# 运行测试
if __name__ == '__main__':
    unittest.main()