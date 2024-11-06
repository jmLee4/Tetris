import simpy
import sys
sys.path.append('..')
from cluster import Cluster
from scheduler import Scheduler

class Simulation(object):
    def __init__(self, configs, algorithm, metric_file, motivation_file, args=None):
        instance_configs, machine_configs, machine_ids, instance_ids = configs
        # 为什么原代码把instance缩写成inc、把machine缩写成mac？这
        self.sand = False
        
        if args is not None:
            # self.drl = args.drl
            self.sand = args.sandpiper

        self.env = simpy.Environment()
        self.cluster = Cluster()
        self.cluster.add_old_new(machine_ids, instance_ids)
        self.cluster.configure_machines(machine_configs)
        self.cluster.configure_instances(instance_configs)
        
        self.scheduler = Scheduler(self.env, algorithm, self.sand, metric_file, motivation_file)
        self.scheduler.attach(self)
    
    def run(self):
        self.env.process(self.scheduler.run())
        print("Start simulation")
        self.env.run()

    def finished(self, clock):
        # 报错：KeyError 0，并不存在ID为0的Instance
        # 不理解这里为什么要选取ID为0的Instance，最早？而且为什么要-2
        if clock >= len(self.cluster.instances[0].cpu_list)-2:
            return True
        return False
 