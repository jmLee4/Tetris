import csv
import numpy as np
from cluster import Cluster
from time import time
from pyDOE import lhs
from abc import ABC, abstractmethod

class Algorithm(ABC):
    @abstractmethod
    def __call__(self, *args):
        pass

class AlgorithmTetris(Algorithm):
    def __call__(self, cluster:Cluster, timeslot, motivation_file):

        self.motivation_file = motivation_file
        self.cluster = cluster

        # 默认参数
        self.params = {
            "w": 6,             # 窗口大小(Time Window Size)，论文中是设置为2
            "z": range(20),     # 采样次数(Number of Samples)
            "k": 5,             # 尝试次数(Number of Trials)
            "v": 0.4,           # 采样率(Sampling Ratio)

            "a": 0.004,         # Alpha，α，迁移代价相比于集群负载不均衡度的归一化参数，见公式(4)
            "b": 0.0025,        # Beta，β，内存资源相比于CPU资源的归一化参数，见公式(1)
            "u": 0.8,           # 疑似是 Upsilon，υ，论文中是设置为10，迁移代价 = u + y * 内存资源，见公式(3)
            "y": 0.25,          # 疑似是 Gamma，γ，论文中是设置为1，迁移代价 = u + y * 内存资源，见公式(3)

            "N": len(self.cluster.instances),    # Instance 数量，即容器数量
            "M": len(self.cluster.machines),     # Machine 数量，即节点(Node)数量
        }
        
        if timeslot == 0:
            with open(motivation_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "container", "metric"])
        value, eval_bal, eval_mig = self.schedule(timeslot)
        
        return value, eval_bal, eval_mig

    # 在特定的时间片(timeslot)调度容器
    def schedule(self, timeslot):
        start_time = time()
        print(f"开始时间片(Timeslot)第 {timeslot} 调度")
        params = self.params
        min_z_record, eval_bal, eval_mig, value = self.tetris_schedule(params["z"], params["k"], params["w"],
                                                                       params["v"], params["M"], params["a"],
                                                                       params["b"], params["y"], timeslot)
        finish_time = time()
        
        if min_z_record != -1:
            print(f"Timeslot {timeslot}，在第 {min_z_record} 次迭代中找到更优的迁移方案，耗时 {finish_time - start_time:.2f}s，"
                  f"负载均衡成本 {eval_bal:.2f}，迁移成本 {eval_mig:.2f}，总成本 {value:.2f}")
        else:
            print(f"Timeslot {timeslot}，没有更优的迁移方案，耗时 {finish_time - start_time:.2f}s，"
                  f"负载均衡成本 {eval_bal:.2f}，迁移成本 {eval_mig:.2f}，总成本 {value:.2f}")
        return value, eval_bal, eval_mig

    # Tetris调度算法，对应Algo.1
    def tetris_schedule(self, Z, K, W, v, M, a, b, y, timeslot):
        cluster = self.cluster
        lenx = len(cluster.instances[0].cpu_list)
        mark_time = time()
        cluster_cost, balfirst = cluster.calculate_cluster_cost(timeslot, W, b)
        print("算法执行，计算 cluster_cost 消耗 %.2fs, cluster_cost = %.3f" % (time()-mark_time, cluster_cost))

        # Phase Ⅰ: Server Classification
        mark_time = time()
        imbalanced_machines = self.server_classification(cluster, b, y, 0)
        print("算法执行，计算 imbalanced_machines 消耗了 %.2fs" % (time()-mark_time))
        print("待迁出Node数量：%d，待迁入Node数量：%d" % (len(imbalanced_machines[0]), len(imbalanced_machines[1])))

        # Phase Ⅱ: Container Scheduling
        candidate_copy = {}
        # min_z_record: Cose最小对应的采样次数，初始为-1，如果返回时还是-1，说明没有找到更优解
        min_z_record, balf, migf, valuef = -1, balfirst, 0, balfirst
        print(f"当前集群Cost: {cluster_cost}，开始采样计算")
        for z in Z:
            cost = 0
            bal, mig, value = 0, 0, 0
            CPU_t, MEM_t = None, None

            for t in range(W):
                if t == 1 and timeslot+W-1 >= lenx:
                    break
                candidate = {}
                
                if t != 0 and flag:
                    imbalanced_machines = self.server_classification(cluster, b, y, t)
                flag = False
                
                for _ in range(K):
                    CPU_t, MEM_t, flag = self.RandomGreedySimplify_new(M, a, b, v, t, imbalanced_machines, candidate, CPU_t, MEM_t)
                    
                    if flag:
                        break
                    candidate.clear()
                
                if t == 0:
                    candidate_copy[z] = candidate
                if len(candidate) == 0 or flag == False:
                    cost += cluster.NoMigration(t)
                    
                    if t == 0:
                        mig, bal, value = 0, cost, cost
                    continue
                
                migx, balx, valuex = cluster.calculate_migrate_cost(candidate, timeslot, t, W, b, a, M)
                cost += valuex
                
                if t == 0:
                    mig, bal, value = migx, balx, valuex
                
                if cost > cluster_cost:
                    # cost 超过了 cluster_cost，提前结束
                    break

            print("[采样轮次Z=%d] Cost计算消耗%.2fs，当前Cost: %.3f" % (z, time()-mark_time, cost))
            if cost < cluster_cost:
                # 更新 cluster_cost
                cluster_cost = cost
                min_z_record, balf, migf, valuef = z, bal, mig, value
            cluster.backZero(z, timeslot, W)

        # 基于计算结果进行Instance的调度
        motivation = cluster.remap_instance_to_machine(candidate_copy, min_z_record, timeslot, W, b)
        print(f"Motivation: {motivation}")
        if len(motivation) > 0:
            motivationFile = self.motivation_file
            
            with open(motivationFile, "a") as f:
                writer = csv.writer(f)
                
                for containerId, metric in motivation.items():
                    writer.writerow([timeslot, containerId, metric])

        return min_z_record, balf, migf, valuef

    """
    原函数名是 findOverAndUnder，我理解对应Algo.1中的 server classification 阶段
    找到一个上界和下界，进而可以划分出2个Node集合：待迁出Node集合和待迁入Node集合
        :param cluster: Cluster对象
        :param b:  Beta，β，内存资源相比于CPU资源的归一化参数，见公式(1)
        :param y:  归一化参数，暂时不了解作用，未在论文中体现
        :param timeslot: 时间片
        
        :returns: migrate_out_machines, migrate_in_machines
        :returns: 超过阈值的Machine列表（待迁出），低于阈值的Machine列表（待迁入）
    """
    def server_classification(self, cluster:Cluster, b, y, timeslot):
        params = self.params

        # 获取t时刻的CPU和MEM数据
        machine_cpu_data = {k: cpu_sum_list[timeslot] for k, cpu_sum_list in cluster.sum_of_cpu.items()}
        machine_mem_data = {k: mem_sum_list[timeslot] for k, mem_sum_list in cluster.sum_of_mem.items()}
        machine_cpu_data = np.array(list(machine_cpu_data.values()))
        machine_mem_data = np.array(list(machine_mem_data.values()))

        # 计算所有Machine的CPU和MEM的平均值和最大值
        avg_cpu = np.sum(machine_cpu_data) / params["M"]
        avg_mem = np.sum(machine_mem_data) / params["M"]
        max_cpu = max(machine_cpu_data)
        max_mem = max(machine_mem_data)

        # 这里是什么，论文中有涉及任何对max的利用吗？
        threshold_cpu = y * (max_cpu - avg_cpu) + avg_cpu
        threshold_mem = y * (max_mem - avg_mem) + avg_mem

        # 迁出的阈值，即哪些Node会被选作SrcNode，对应公式(12)，似乎跟论文中的描述不太一样
        threshold_src = (threshold_cpu ** 2 + b * threshold_mem ** 2) / 2
        # 迁入的阈值，即哪些Node会被选作DestNode，对应公式(13)的一部分，下面会对其进行修正
        threshold_dest = (avg_cpu ** 2 + b * avg_mem ** 2) / 2

        # 补充一下上面的内容
        # 原代码是：thresh_out = (thr_CPU ** 2 + b * thr_MEM ** 2) / 2
        #          thresh_in = (avg_CPU ** 2 + b * avg_MEM ** 2) / 2
        # 我理解 thresh_in 是待迁入，所以对应的是 threshold_dest；同理有 thresh_out 对应 threshold_src；之前弄反了

        cpu_t = cluster.instance_cpu[:, timeslot]
        mem_t = cluster.instance_mem[:, timeslot]
        cpu_mem = np.vstack((cpu_t, mem_t)).T
        cpu_mem_sorted = cpu_mem[np.lexsort(cpu_mem[:, ::-1].T)]    # 按CPU和内存降序排序

        cpu_sum = 0
        mem_sum = 0

        # 对应公式(13)的另一部分，修正迁入阈值
        for k in cpu_mem_sorted:
            cpu_sum = cpu_sum + k[0]
            mem_sum = mem_sum + k[1]
            
            if cpu_sum < avg_cpu and mem_sum < avg_mem:
                temp_value = (k[0] ** 2 + b * k[1] ** 2) / 2
                threshold_dest = threshold_dest - temp_value
            else:
                temp_value = ((avg_cpu - cpu_sum + k[0]) ** 2 + b * (avg_mem - mem_sum + k[1]) ** 2) / 2
                threshold_dest = threshold_dest - temp_value
                break

        # 至今理解不了CsPluMs的含义，原代码是：allVmCsPluMs = cluster.pm_cost，明显混用了 vm 和 pm 的概念
        allVmCsPluMs = cluster.machine_cost

        machine_id_2_load_imbalance_degree = {machine_id: v[timeslot] for machine_id, v in allVmCsPluMs.items()}
        machine_ids = np.array(list(machine_id_2_load_imbalance_degree.keys()))
        load_imbalance_degree = np.array(list(machine_id_2_load_imbalance_degree.values()))
        # 负载不均衡度大于迁出阈值的Machine，即待迁出Machine
        load_over_threshold = np.where(load_imbalance_degree > threshold_src)[0]
        migrate_out_machines = machine_ids[load_over_threshold]
        # 负载不均衡度小于迁入阈值的Machine，即待迁入Machine
        load_under_threshold = np.where(load_imbalance_degree < threshold_dest)[0]
        migrate_in_machines = machine_ids[load_under_threshold]
        
        return migrate_out_machines, migrate_in_machines

    def RandomGreedySimplify_new(self, M, a, b, v, t, findOV, candidate, CPU_t=None, MEM_t=None):
        cluster = self.cluster
        machines = cluster.machines
        cpusum = cluster.sum_of_cpu
        memsum = cluster.sum_of_mem

        cpu_t = list(cluster.instance_cpu[:, t])
        mem_t = list(cluster.instance_mem[:, t])

        over, under = findOV
        
        if CPU_t is None or MEM_t is None:
            CPU_t = {k: cpusumlist[t] for k, cpusumlist in sorted(
                cpusum.items(), key=lambda x: x[0])}
            MEM_t = {k: memsumlist[t] for k, memsumlist in sorted(
                memsum.items(), key=lambda x: x[0])}
            CPU_t = np.array(list(CPU_t.values()))
            MEM_t = np.array(list(MEM_t.values()))

        for s in over:
            machinethis = machines[s]
            instances = machinethis.instances
            mig_candi_s = np.array([x for x in instances.keys()])

            samples = np.ceil(v*len(mig_candi_s))
            samples = int(samples)
            lhd = lhs(1, samples)
            mig_loc = lhd * len(mig_candi_s)
            mig_loc = mig_loc[:, 0].astype(int)

            mig = np.unique(mig_candi_s[mig_loc]).astype(int)
            
            for m in mig:
                destination = s
                m = int(m)

                bal_d_cpu = cpu_t[m] * (CPU_t[s] - cpu_t[m] - CPU_t[under])
                bal_d_mem = mem_t[m] * (MEM_t[s] - mem_t[m] - MEM_t[under])

                bal_d = np.array(bal_d_cpu + b * bal_d_mem)
                mig_m = np.array(a * (M-1) * mem_t[m])
                idx = np.array(np.where(bal_d > mig_m)[0])
                lendx = len(idx)
                
                if lendx == 0:
                    continue
                allmetric = bal_d

                tmps = {under[idx[i]]: allmetric[idx[i]] for i in range(lendx)}
                candiUnder = [k for k, v in sorted(
                    tmps.items(), key=lambda x:x[1], reverse=True)]

                for destination in candiUnder:
                    rescpu = CPU_t[destination]+cpu_t[m]
                    resmem = MEM_t[destination]+mem_t[m]
                    
                    if destination != s and \
                            rescpu < machinethis.cpu_capacity and \
                            resmem < machinethis.mem_capacity:
                        CPU_t[s] -= cpu_t[m]
                        CPU_t[destination] = rescpu
                        MEM_t[s] -= mem_t[m]
                        MEM_t[destination] = resmem
                        
                        if m not in candidate:
                            candidate[m] = [(s, destination)]
                        else:
                            candidate[m].append((s, destination))
                        break

        if len(candidate) > 0:
            for k in machines.keys():
                if machines[k].cpu_capacity < CPU_t[k] or machines[k].mem_capacity < MEM_t[k]:
                    return None, None, False
        
        if len(candidate) <= 0:
            return None, None, False
        
        return CPU_t, MEM_t, True
