import argparse
import os
import warnings
import inspect
import builtins

from algorithm import AlgorithmTetris
from simulation import Simulation

# Ignore specific warnings (pmdarima)
warnings.filterwarnings("ignore", message="Input time-series is completely constant; returning a (0, 0, 0) ARMA.")
warnings.filterwarnings("ignore", category=UserWarning)

# Print debug information
original_print = builtins.print

def custom_print(*args, **kwargs):
    frame = inspect.currentframe().f_back
    file_name = os.path.relpath(frame.f_code.co_filename)
    line_number = frame.f_lineno
    original_print(f"[{file_name}:{line_number}]", *args, **kwargs)

builtins.print = custom_print

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="manual to this script")
    parser.add_argument("--arima", type=bool, default=False)
    parser.add_argument("--sandpiper", type=bool, default=False)
    args = parser.parse_args()

    filepath = r"../simulation_data/target"

    metric_log_filename = "metric"

    from data.loader_reduce import load_instance_data
    
    # test_array = [[3989, 67437], [100, 1700], [500, 8500], [1000, 17000], [2000, 34000], [3000, 51000]]
    # test_array [[节点数, 容器数], ...]
    test_array = [[10, 170]]
    # configs是一个List，每个test_array对应一组[new_instances, new_machines, machine_ids, instance_ids]
    # new_instances是一个字典，key是instance_id，value是InstanceConfig对象
    # instance_ids是一个字典，是新ID和旧ID的映射
    configs = load_instance_data(filepath, test_array)

    algorithm = AlgorithmTetris()
    
    for i, config in enumerate(configs):

        res_struct_filename = os.path.join(os.getcwd(), str(len(config[1])) + "-struct.json")
        metricFile = os.path.join(os.getcwd(), str(len(config[1])) + "-metric.csv")
        motivationFile = os.path.join(os.getcwd(), str(len(config[1])) + "-motivation.csv")

        print("############################# 第", i+1, "组 #############################")
        
        simulator = Simulation(config, algorithm, metricFile, motivationFile, args)
        simulator.run()

"""
【注意事项】
    1. 代码变量命名的时候有涉及到 vm 的概念，初步看指的是 Instance
    2. 代码中还有 pm 的概念，我怀疑早期被用于虚拟机放置，所以 vm 对应 Virtual Machine、pm 对应 Physical Machine
       映射过来就是：vm 对应 Instance、pm 对应 Machine
【总结】
    1. loader_reduce.py 加载数据集，生成Machine和Instance的配置信息
    2. simulation.py 模拟器类，初始化Cluster和Scheduler，运行模拟器
    2.1 创建Cluster对象，添加机器和实例，配置机器和实例
    2.2 创建Scheduler对象，数据记录到 metric_file 和 motivation_file，运行调度器
"""
