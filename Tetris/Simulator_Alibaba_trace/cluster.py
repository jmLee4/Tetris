import time
from typing import Dict
import numpy as np
from machine import Machine
from instance import Instance

class Cluster(object):
    def __init__(self):
        self.machines = {}
        self.instances = {}  
        
        self.sum_of_cpu = None
        self.sum_of_mem = None
        self.sum_of_cpu_copy = None
        self.sum_of_mem_copy = None
        
        self.instance_cpu = None
        self.instance_mem = None
        
        self.machine_cost = None
        self.pm_cost_copy = None
        self.t_pm_cost_motivatin = {}

        self.modifyPmCopy=None
        self.driftPm = {}
        
        
        self.over = None
        self.under = None
        self.candidate_machine = None
        self.candidate_container = None
        self.candidate_finanlly = []
        self.oldmac_containers = {}
        
    
    def attachOverUnder(self,over,under):
        x = len(over) 
        y =  len(under)
        
        self.all_candidate_c = {}
        self.over= over[:x]
        self.candidate_machine = { k: self.machines[k] for k in self.over }
        self.under = under[:y]
        self.candidate_container = {  k:self.instances[k] for k in self.under}
        
    
    def add_old_new(self, mac_ids, inc_ids):
        self.mac_ids = mac_ids
        self.inc_ids = inc_ids
  
    
    def copy_machine(self,machines):
        self.machines= machines.copy()
    
    
    def copy_instances(self,instances):
        self.instances = instances.copy()
    
    
    def configure_machines(self, machine_configs: dict):
        for machine_config in machine_configs.values():
            machine = Machine(machine_config)
            self.machines[machine.id] = machine
            machine.attach(self)

    
    def configure_instances(self, instance_configs: dict):
        for instance_config in instance_configs.values():
            inc = Instance(instance_config)
            self.instances[inc.id] = inc
            
            machine_id = inc.machine_id
            machine = self.machines.get(machine_id, None)
           
            assert machine is not None
            machine.push(inc)
    
    
    def cost_all_pm(self,clock,w,b):
        machines = self.machines
        cost_min = 0
        
        for pm in machines.values(): 
            v = pm.calculate_cost(clock, w, b)
           
            cost_min += v
        return cost_min
    
    # 计算所有节点(整个集群)的成本
    # 参考Ⅲ.A，集群的成本等于Server上每种资源的负载不均衡度，先不考虑迁移成本
    def calculate_cluster_cost(self, clock, w, b):

        sum_of_cpu = self.sum_of_cpu = {}
        sum_of_mem = self.sum_of_mem = {}
        instance_cpu_record = {}
        instance_mem_record = {}
        machine_cost = {}
        machines = self.machines

        sum_of_cost, balance = 0, 0
        for machine in machines.values():
            cost = machine.getnowPluPredictCost(clock, w, b)
            machine_cost[machine.id] = machine.CsPluMs
            
            try:
                balance += machine_cost[machine.id][0]
            except:
                balance += 0
                machine_cost[machine.id] = np.array([0 for _ in range(w)])
            
            sum_of_cpu[machine.id] = machine.cpu_sum_w
            sum_of_mem[machine.id] = machine.mem_sum_w
            instance_cpu_record.update(machine.cpuPluPredict)
            instance_mem_record.update(machine.memPluPredict)
            
            sum_of_cost += cost
        
        instance_cpu_record = sorted(instance_cpu_record.items(),key=lambda x:x[0])
        instance_mem_record = sorted(instance_mem_record.items(),key=lambda x:x[0])
        
        self.instance_cpu = np.array([v for k, v in instance_cpu_record])
        self.instance_mem = np.array([v for k, v in instance_mem_record])
        
        print(len(instance_cpu_record),len(self.instances),len(instance_cpu_record))
        assert len(self.instance_cpu) == len(self.instances)
        self.machine_cost = machine_cost
        self.pm_cost_copy = {k:v for k,v in machine_cost.items() }
        self.sum_of_cpu_copy = {k:v for k,v in sum_of_cpu .items()}
        self.sum_of_mem_copy = {k:v for k,v in sum_of_mem .items()}
        self.modifyPmCopy = []
        self.t_pm_cost_motivatin[clock] = {k:v for k,v in machine_cost.items() }
        
        return sum_of_cost, balance
    
    # 计算迁移代价
    def calculate_migrate_cost(self, candidate:Dict, timeslot, t, w, b, a, M):
        machines = self.machines
        instances = self.instances
        mig = 0
        bal = 0
        mac_modify = set()
        cpusum = self.sum_of_cpu
        memsum = self.sum_of_mem
        pm_cost = self.machine_cost
          
        candidate_s_d = [ x for v in list(candidate.values()) for x in v[-1]]
        mac_modify.update(candidate_s_d)
        otherPmCost = [ v[t] for k,v in pm_cost.items() if k not in mac_modify ]
        
        assert len(otherPmCost) + len(mac_modify) == len(machines)
        otherPmCostSum = np.sum( otherPmCost )
        
        for vmid,slou in candidate.items():
            s,destination=slou[-1]
            mig += machines[s].migrateOut(vmid,t)
            machines[destination].migrateIn(instances[vmid],t)
        
        for macid in mac_modify:
            bal+=machines[macid].afterMigration_cost(timeslot, t, w, b)
            cpusum[macid] = machines[macid].cpu_sum_w
            memsum[macid] = machines[macid].mem_sum_w
            pm_cost[macid] = machines[macid].CsPluMs_migraton 
        
        self.modifyPmCopy.append(candidate)
        return mig,bal+otherPmCostSum,bal+(M-1)*mig*a+otherPmCostSum
    
    
    def NoMigration(self,t):
        pm_cost = self.machine_cost
        PmCost = np.sum([ v[t] for k,v in pm_cost.items() ])
        
        return PmCost
    
    
    def backZero(self,z,clock,w):
        cpusum_copy = self.sum_of_cpu_copy
        memsum_copy = self.sum_of_mem_copy
        pm_cost_copy = self.pm_cost_copy
        self.pm_cost_copy = {k:v for k,v in pm_cost_copy.items() }
        self.sum_of_cpu_copy = {k:v for k,v in cpusum_copy.items()}
        self.sum_of_mem_copy = {k:v for k,v in memsum_copy.items()}
        
        machines = self.machines
        instances = self.instances
        self.machine_cost = {k:v for k,v in pm_cost_copy.items()}
        self.sum_of_cpu = {k:v for k,v in cpusum_copy.items()}
        self.sum_of_mem = {k:v for k,v in memsum_copy.items()}
        
        print(f"len of modifyPmCopy: {len(self.modifyPmCopy)}")
        macids = set()
        
        if len(self.modifyPmCopy) > 0:
            x = range(len(self.modifyPmCopy)-1,-1,-1)
            
            for i in x:
                candidate = self.modifyPmCopy[i]
                
                for vmid,v in candidate.items():
                    destination,s = v[0]
                    machines[s].pop(vmid)
                    machines[destination].push(instances[vmid])
                    macids.add(s)
                    macids.add(destination)
        
        for macid in macids:
            machines[macid].getEveryTimeCpuList(clock,w)
        
        self.modifyPmCopy = []
        
    
    # def freshStructPmVm(self,candidate_copy,z,clock,w,b):
    #     if z==-1:
    #         return {}
        
    #     candidate = candidate_copy[z]
        
    #     if len(candidate) == 0:
    #         print("no migration")
    #         return {}
        
    #     machines = self.machines
    #     instances = self.instances
    #     outpm = {v[0][0]:0 for k,v in candidate.items() }
    #     outpm = {macid:set([v for v in machines[macid].instances.keys()])for macid in outpm.keys()}
    #     inpm = {v[0][1]:0 for k,v in candidate.items() }
    #     inpm = {macid:set([v for v in machines[macid].instances.keys()])for macid in inpm.keys()}
    #     print(f'outpm is {outpm}')
    #     print(f'inpm is {inpm}')
    #     motivation = {}
    #     moti_len = 2
        
    #     for vmid,v in candidate.items():
    #         if (len(v)>1):
    #             print(v)
    #             assert 1==0
            
    #         s,destination = v[0]
    #         before_value = []
    #         after_value = []
            
    #         # before
    #         try:
    #             for t in range(moti_len):
    #                 beforePmOutCost = machines[s].afterOneContainerMigration(clock+t,w,b)
    #                 beforePmInCost = machines[destination].afterOneContainerMigration(clock+t,w,b)
    #                 before_value.append(beforePmOutCost+beforePmInCost)
    #                 print(f"计算 before_value 耗时 {time() - start_time:.2f} 秒")
    #         except Exception as e:
    #             print(f"计算 before_value 出错: {e}")
    #         machines[s].pop(vmid)
    #         machines[destination].push(instances[vmid])
            
    #         try:
    #             for t in range(moti_len):
    #                 afterPmOutCost = machines[s].afterOneContainerMigration(clock+t,w,b)
    #                 afterPmInCost = machines[destination].afterOneContainerMigration(clock+t,w,b)
    #                 after_value.append(afterPmOutCost+afterPmInCost)
    #         except:
    #             print()
            
    #         motivation[vmid] = [s,destination,before_value,after_value]  
        
    #     afteroutpm = {macid:set([v for v in machines[macid].instances.keys()])for macid in outpm.keys()}
    #     afterinpm = {macid:set([v for v in machines[macid].instances.keys()])for macid in inpm.keys()}
        
    #     diffout = {macid:v.difference(afteroutpm[macid])for macid,v in outpm.items()}
    #     diffin = {macid:afterinpm[macid].difference(v)for macid,v in inpm.items()}
        
    #     violations = self.isAllUnderLoad(clock)
        
    #     self.driftPm[clock]={"outpm":outpm,"afteroutpm":afteroutpm,"inpm":inpm,"afterinpm":afterinpm,"diffout":diffout,"diffin":diffin,"violations":violations}
    #     return motivation

    """
    原函数名是 freshStructPmVm，替换 pm 为 machine、vm 为 instance
    所谓的 freshStruct 是修改结构体，我理解是设置 instance 和 machine_id、设置 machine 拥有的 instance_ids，等价于调度
    """

    def remap_instance_to_machine(self, candidate_copy, z, clock, w, b):
        if z == -1:
            return {}

        candidate = candidate_copy[z]
        if len(candidate) == 0:
            print("no migration")
            return {}

        from time import time

        machines = self.machines
        instances = self.instances

        outpm = {v[0][0]: 0 for k, v in candidate.items()}
        outpm = {macid: set([v for v in machines[macid].instances.keys()]) for macid in outpm.keys()}
        inpm = {v[0][1]: 0 for k, v in candidate.items()}
        inpm = {macid: set([v for v in machines[macid].instances.keys()]) for macid in inpm.keys()}
        print(f'outpm is {outpm}')
        print(f'inpm is {inpm}')

        motivation = {}
        moti_len = 2

        for vmid, v in candidate.items():
            if len(v) > 1:
                print(v)
                assert 1 == 0

            s, destination = v[0]
            before_value = []
            after_value = []

            # before
            try:
                start_time = time()
                for t in range(moti_len):
                    beforePmOutCost = machines[s].afterOneContainerMigration(clock + t, w, b)
                    beforePmInCost = machines[destination].afterOneContainerMigration(clock + t, w, b)
                    before_value.append(beforePmOutCost + beforePmInCost)
                print(f"计算 before_value 耗时 {time() - start_time:.2f} 秒")
            except Exception as e:
                print(f"计算 before_value 出错: {e}")

            start_time = time()
            machines[s].pop(vmid)
            machines[destination].push(instances[vmid])
            print(f"迁移容器 {vmid} 耗时 {time() - start_time:.2f} 秒")

            try:
                start_time = time()
                for t in range(moti_len):
                    afterPmOutCost = machines[s].afterOneContainerMigration(clock + t, w, b)
                    afterPmInCost = machines[destination].afterOneContainerMigration(clock + t, w, b)
                    after_value.append(afterPmOutCost + afterPmInCost)
                print(f"计算 after_value 耗时 {time() - start_time:.2f} 秒")
            except Exception as e:
                print(f"计算 after_value 出错: {e}")

            motivation[vmid] = [s, destination, before_value, after_value]

        afteroutpm = {macid: set([v for v in machines[macid].instances.keys()]) for macid in outpm.keys()}
        afterinpm = {macid: set([v for v in machines[macid].instances.keys()]) for macid in inpm.keys()}

        diffout = {macid: v.difference(afteroutpm[macid]) for macid, v in outpm.items()}
        diffin = {macid: afterinpm[macid].difference(v) for macid, v in inpm.items()}

        violations = self.isAllUnderLoad(clock)

        self.driftPm[clock] = {
            "outpm": outpm, "afteroutpm": afteroutpm, "inpm": inpm,
            "afterinpm": afterinpm, "diffout": diffout, "diffin": diffin,
            "violations": violations
        }

        return motivation

    def isAllUnderLoad(self,clock,sand=False):
        machines = self.machines
        cpusum = self.sum_of_cpu
        memsum = self.sum_of_mem
        cpu_capacity = 20 if sand else 30
        
        try:
            violations = {mac.id:machines[mac.id].instances.keys() for mac in machines.values() if cpusum[mac.id][0] >cpu_capacity or memsum[mac.id][0]>mac.mem_capacity }
        except:
            violations = {}
            print("wrong")
           
        return violations
    
    def plt(self,outpm,afteroutpm,clock):
        print("#"*30)
        
        for k in outpm.keys():
            res = "\t"*5+"pm["+str(k)+"]\n"
            res += "-"*20
        pass
   
    
    @property
    def drift_json(self):
        return [
             {
                'time': time.asctime(time.localtime(time.time()))
            },
             {
                 k:{
                    names:
                      {str(kv):str(vv) 
                       for kv,vv in value.items()
                       }
                     for names,value in v.items()
                     }
                 for k,v in self.driftPm.items()
             }
            
        ]
        
    
    @property
    def structure(self):
        return [
            {
                'time': time.asctime(time.localtime(time.time()))
            },

            {

                i: {
                    'cpu_capacity': m.cpu_capacity,
                    'memory_capacity': m.mem_capacity,
                  
                    'cpu': m.cpu,
                    'memory': m.mem,
                   
                    'instances': {
                        j: {
                            'cpu': inst.cpu,
                            'mem': m.mem,
                        } for j, inst in m.instances.items()
                    }
                }
                for i, m in self.machines.items()
            }]
