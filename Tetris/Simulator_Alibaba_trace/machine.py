from instance import Instance
import numpy as np


class MachineConfig(object):
    def __init__(self, machine_id, cpu_capacity, memory_capacity):
        self.id = machine_id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity


class Machine(object):
    def __init__(self, machine_config):
        self.id = machine_config.id
        self.cpu_capacity = machine_config.cpu_capacity
        self.mem_capacity = machine_config.memory_capacity
        self.isMigration = False
        self.cluster = None
        self.instances_bak = None
        self.instances = {}
        self.len = 0

        self.cpu_plus_predict = None
        self.mem_plus_predict = None

        self.cost_record_plus_predict_value = {}

        self.cpu_sum_w = None
        self.mem_sum_w = None

        self.weighted_cpu_mem_sum = None
        self.CsPluMs_migraton = None

    # 将机器对象与集群（Cluster）关联起来
    def attach(self, cluster):
        self.cluster = cluster

    # 向机器中添加实例（Instance）
    def add_instance(self, instance_config):

        instance = Instance(instance_config)
        instance_config.machine_id = self.id
        instance.attach(self)
        self.instances[instance.id] = instance

    # 从机器中移除指定id的实例，并返回
    def pop(self, instance_id):
        instance = self.instances.pop(instance_id)
        instance.machine = None
        return instance

    # 将实例对象添加到机器中
    def push(self, instance):
        self.instances[instance.id] = instance
        instance.attach(self)

    # 根据时间窗口 w 大小，获取Machine中每个Instance的资源使用+预测情况列表
    def calculate_predict_value_at_clock(self, clock, w):
        instances = self.instances
        self.len = len(instances)
        self.cpu_plus_predict = {v.id: np.array(v.cpu_list[clock:clock + 1] + v.predict(clock, w)[clock]["cpu"]) for v in instances.values()}
        self.mem_plus_predict = {v.id: np.array(v.mem_list[clock:clock + 1] + v.predict_result[clock]["mem"]) for v in instances.values()}

    # 计算此刻(clock)，Machine中所有Instance的CPU使用总和和内存使用总和
    def calculate_cpu_sum_and_mem_sum(self):
        self.cpu_sum_w = np.sum(np.array([v for k, v in self.cpu_plus_predict.items()]), axis=0)
        self.mem_sum_w = np.sum(np.array([v for k, v in self.mem_plus_predict.items()]), axis=0)

    # 计算Machine在指定时间范围内的成本
    def calculate_cost(self, clock, w, b=0.0025):
        self.calculate_predict_value_at_clock(clock, w)
        self.calculate_cpu_sum_and_mem_sum()

        instance_cpu = np.array([v for k, v in self.cpu_plus_predict.items()])
        instance_mem = np.array([v for k, v in self.mem_plus_predict.items()])
        weighted_cpu_mem_sum = [0.0 for _ in range(w)]   # 原变量命名为 csplums，按照对命名的理解，c 是 CPU，m 是 Memory，plu 是 Plus

        for i in range(len(instance_cpu)-1):
            for t in range(w):
                c = instance_cpu[i][t] * np.sum(instance_cpu[i+1:, t])
                m = instance_mem[i][t] * np.sum(instance_mem[i+1:, t])
                # 不太像公式(1)，没有减去平均值，只是单纯地聚合了CPU和Mem两种资源的使用情况
                weighted_cpu_mem_sum[t] += c + b * m

        # 首先这个循环是可以合并到上面去的；其次减去0.5是为了归一化？csplums 的取值原先只有[0, 1]吗？
        for t in range(w):
            if weighted_cpu_mem_sum[t] > 0.5:
                weighted_cpu_mem_sum[t] -= 0.5
        machine_cost = np.sum(weighted_cpu_mem_sum)

        self.weighted_cpu_mem_sum = weighted_cpu_mem_sum
        self.cost_record_plus_predict_value[clock] = machine_cost

        return weighted_cpu_mem_sum[0]

    # 获取指定时间点的成本值
    def get_cost_plus_predict_value(self, clock, w, b):
        # 存在缓存机制，如果已经计算过，则直接返回
        if clock not in self.cost_record_plus_predict_value or self.cost_record_plus_predict_value[clock] is None:
            self.calculate_cost(clock, w, b)
        return self.cost_record_plus_predict_value[clock]

    # 将指定id的实例从机器中迁出，返回迁出操作内存消耗
    def migrateOut(self, vmid, t):
        self.pop(vmid)
        return self.mem_plus_predict[vmid][t]*2

    # 将指定id的实例迁入到机器中
    def migrateIn(self, vmid, t):
        self.push(vmid)

    # 计算迁移后指定时间点的成本
    def afterMigration_cost(self, clock, t, w, b):

        cost_t = {}
        cpu_vm = np.array([v for k, v in self.cpu_plus_predict.items()])
        mem_vm = np.array([v for k, v in self.mem_plus_predict.items()])

        csplums = [0.0 for i in range(w)]
        
        for i in range(len(cpu_vm)-1):
            for t in range(w):
                c = cpu_vm[i][t] * np.sum(cpu_vm[i+1:, t])
                m = mem_vm[i][t] * np.sum(mem_vm[i+1:, t])
                csplums[t] += c + b * m

        cost_t[t] = csplums[t]

        self.CsPluMs_migraton = csplums
        return cost_t[t]

    # 计算执行一次容器迁移后的成本
    def afterOneContainerMigration(self, clock, w, b):
        self.calculate_predict_value_at_clock(clock, w)
        self.calculate_cpu_sum_and_mem_sum()
        csplums = []
        pm_cost = 0
        cpu_vm = np.array([v for k, v in self.cpu_plus_predict.items()])
        mem_vm = np.array([v for k, v in self.mem_plus_predict.items()])
        csplums = [0.0 for i in range(w)]
        pm_cost = 0
        
        for i in range(len(cpu_vm)-1):
            for t in range(w):
                c = cpu_vm[i][t] * np.sum(cpu_vm[i+1:, t])
                m = mem_vm[i][t] * np.sum(mem_vm[i+1:, t])
                csplums[t] += c + b * m
        
        for t in range(w):
            if csplums[t] > 0.5:
                csplums[t] -= 0.5
        
        pm_cost = np.sum(csplums)
        
        return pm_cost

    # 返回机器中剩余的CPU容量
    @property
    def cpu(self):
        occupied = 0
        
        for instance in self.instances.values():
            occupied += instance.cpu
        
        return self.cpu_capacity - occupied

    # 返回机器中剩余的内存容量
    @property
    def mem(self):
        occupied = 0
        
        for instance in self.instances.values():
            occupied += instance.mem

        return self.mem_capacity - occupied

    
    @property
    def cpusum(self):
        occupied = 0
        
        for instance in self.instances.values():
            occupied += instance.cpu
        
        return self.cpu_capacity - occupied

    
    @property
    def memsum(self):
        occupied = 0
        
        for instance in self.instances.values():
            occupied += instance.mem

        return self.mem_capacity - occupied