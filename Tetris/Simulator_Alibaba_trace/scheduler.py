from time import time
from scipy.stats import qmc
import csv
import os

class Scheduler(object):
    def __init__(self, env, algorithm, sand, metric_file, motivation_file):

        # 通过Attach的方式设置
        self.simulation = None
        self.cluster = None

        self.env = env
        self.algorithm = algorithm
        self.dir = os.getcwd()
        self.sand = sand

        self.metric_file = metric_file
        self.motivation_file = motivation_file

        with open(self.motivation_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "container", "metric"])
        with open(self.metric_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["clock", "eval_bal", "eval_mig", "sum", "sums", "time", "total_time", "violation"])

        self.sampler = qmc.LatinHypercube(d=1, seed=42)

    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster
    
    def run(self):
        sums = 0
        alltime = 0
        algorithm = self.algorithm
        
        while not self.simulation.finished(self.env.now):
            start_of_time = time()
            end_of_time = self.env.now

            # timeslot级别的算法调用，sampler需要在这里之前创建
            value, eval_bal, eval_mig = algorithm(self.cluster, self.env.now, self.motivation_file, self.sampler)
            time_used = time() - start_of_time
            print(f"Algorithm execution time: {time_used:.2f}s")

            sums += value
            alltime += time_used

            load_start = time()
            vms = [len(v) for k, v in self.cluster.isAllUnderLoad(self.env.now, self.sand).items()]
            print(f"isAllUnderLoad execution time: {time() - load_start:.2f}s")
            vmlen = sum(vms)

            with open(self.metric_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([end_of_time, eval_bal, eval_mig, value, sums, time_used, alltime, vmlen])
            yield self.env.timeout(1)
        
        print("Now finish time:", self.env.now)


    