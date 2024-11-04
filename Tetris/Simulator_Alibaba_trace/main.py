import argparse
import os
import warnings
import inspect
import builtins

from algorithm import Algorithm_tetris
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

    from data.loader_reduce import InstanceConfigLoader
    
    # test_array = [[3989, 67437], [100, 1700], [500, 8500],
    #               [1000, 17000], [2000, 34000], [3000, 51000]]
    test_array = [[10, 170]]
    configs = InstanceConfigLoader(filepath, test_array) # 这里怀疑是filepath的文件有问题

    algorithm = Algorithm_tetris()
    
    for i, config in enumerate(configs):

        res_struct_filename = os.path.join(os.getcwd(), str(len(config[1])) + "-struct.json")
        metricFile = os.path.join(os.getcwd(), str(len(config[1])) + "-metric.csv")
        motivationFile = os.path.join(os.getcwd(), str(len(config[1])) + "-motivation.csv")

        print("#############################", i+1, "#############################")
        
        simulator = Simulation(config, algorithm, metricFile, motivationFile, args)
        simulator.run()
