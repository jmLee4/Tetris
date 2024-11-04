import pmdarima as pm       

class InstanceConfig(object):
    def __init__(self, machine_id, instance_id, cpu, memory, disk, cpu_curve=None, memory_curve=None):
        self.instance_id = instance_id
        self.machine_id = machine_id
        self.cpu = cpu 
        self.memory = memory
        self.disk = disk
        self.cpu_curve = cpu_curve
        self.memory_curve = memory_curve

class Instance(object):
    def __init__(self, instance_config:InstanceConfig):
        self.id = instance_config.instance_id
        self.machine_id = instance_config.machine_id
        self.cpu = instance_config.cpu
        self.mem = instance_config.memory
        self.mem_list = instance_config.memory_curve.copy()
        self.disk = instance_config.disk
        self.cpu_list = instance_config.cpu_curve.copy()
        self.predict_result = {}
        self.machine = None
        self.predict_cpu_list = []
        self.predict_mem_list = []

    def attach(self, machine):
        self.machine = machine
    
    def predict(self, clock, w, flag=False):
        if flag:
            return self.predict_result
        
        predict_result = self.predict_result
        model = pm.auto_arima(self.cpu_list, start_p=0, start_q=0,
                              information_criterion="aic",
                              test="adf",
                              max_p=3, max_q=3,
                              d=0,              # ADF检验来确定差分阶数d
                              seasonal=True,    # 原本的注释是"No Seasonality"，但是却设置了seasonal=True
                              trace=False,
                              error_action="ignore",
                              suppress_warnings=True,
                              stepwise=True)
        # 模拟实验不涉及到Pearson相关系数和LSTM模型吗？只使用ARIMA模型？
        
        self.predict_cpu_list.append(model.predict(n_periods=w).tolist())
        self.predict_mem_list.append(model.predict(n_periods=w).tolist())
        # Model只传入了cpu_list数据，为什么可以预测mem_list数据？
        
        predict_cpu = self.cpu_list + self.predict_cpu_list[-1]
        predict_mem = self.mem_list + self.predict_mem_list[-1]
        
        self.cpu = predict_cpu[0]
        self.mem = predict_mem[0]
        
        predict_result[clock] = {"cpu":predict_cpu, "mem":predict_mem}
        
        return predict_result
