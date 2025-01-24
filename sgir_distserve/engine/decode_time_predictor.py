class TimePredictor:
    def __init__(self, alpha=0.2, beta=0.2):
        self.alpha = alpha
        self.beta = beta
        self.historic_times = {}
    
    def add_data(self, request_id, decode_time):
        if request_id in self.historic_times:
            self.historic_times[request_id].append(decode_time)
        else:
            self.historic_times[request_id] = [decode_time]
    
    def remove_data(self, request_id):
        if request_id in self.historic_times:
            del self.historic_times[request_id]
    
    def double_exponential_smoothing(self, historic_times):
        if len(historic_times) < 2:
            raise ValueError("需要至少两个数据点来计算二次指数平滑。")
        
        # 初始化平滑值
        level = historic_times[0]
        trend = historic_times[1] - historic_times[0]
        
        # 存储平滑值和趋势值
        levels = [level]
        trends = [trend]
        
        for t in range(1, len(historic_times)):
            level = self.alpha * historic_times[t] + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - levels[-1]) + (1 - self.beta) * trend
            levels.append(level)
            trends.append(trend)
        
        # 预测下一次的运行时间
        next_level = levels[-1] + trends[-1]
        return next_level

