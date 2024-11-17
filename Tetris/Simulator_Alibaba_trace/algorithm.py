import csv
import numpy as np
from cluster import Cluster
from time import time
from pyDOE import lhs
from scipy.stats import qmc
from abc import ABC, abstractmethod

class Algorithm(ABC):
    @abstractmethod
    def __call__(self, *args):
        pass

class AlgorithmTetris(Algorithm):
    def __call__(self, cluster:Cluster, timeslot, motivation_file, sampler):

        self.motivation_file = motivation_file
        self.cluster = cluster
        self.sampler = sampler

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
        print(f"当前执行时间片(Timeslot)：{timeslot}")
        params = self.params
        min_z_record, imbalance_degree_at_t, migrate_cost_at_t, value = self.tetris_schedule(
            params["z"], params["k"], params["w"], params["v"], params["M"], params["a"], params["b"], params["y"], timeslot
        )
        finish_time = time()
        
        if min_z_record != -1:
            print(f"Timeslot {timeslot}，在第 {min_z_record} 次迭代中找到更优的迁移方案，耗时 {finish_time - start_time:.2f}s，"
                  f"负载均衡成本 {imbalance_degree_at_t:.2f}，迁移成本 {migrate_cost_at_t:.2f}，总成本 {value:.2f}")
        else:
            print(f"Timeslot {timeslot}，没有更优的迁移方案，耗时 {finish_time - start_time:.2f}s，"
                  f"负载均衡成本 {imbalance_degree_at_t:.2f}，迁移成本 {migrate_cost_at_t:.2f}，总成本 {value:.2f}")
        return value, imbalance_degree_at_t, migrate_cost_at_t

    # Tetris调度算法，对应Algo.1
    def tetris_schedule(self, Z, K, W, v, M, a, b, y, timeslot):
        cluster = self.cluster

        length_of_cpu_list = len(cluster.instances[0].cpu_list)
        mark_time = time()
        print("算法执行，开始计算 cluster_cost")
        cluster_cost, cluster_imbalance = cluster.calculate_cluster_cost(timeslot, W, b)
        print("算法执行，计算 cluster_cost 消耗 %.2fs, cluster_cost = %.3f" % (time()-mark_time, cluster_cost))

        # Phase Ⅰ: Server Classification

        imbalanced_machines = self.server_classification(cluster, b, y, 0)
        print("待迁出Node数量：%d，待迁入Node数量：%d" % (len(imbalanced_machines[0]), len(imbalanced_machines[1])))

        # Phase Ⅱ: Container Scheduling

        # candidate 本身是个Dict，存储
        # z_2_candidate 是代码之前的逻辑，暂时理解为需要按 z 记录下每次采样的结果
        z_2_candidate = {}
        # min_z_record: Cose最小对应的采样次数，初始为-1，如果返回时还是-1，说明没有找到更优解
        min_z_record, imbalance_cost, migrate_cost, all_cost = -1, cluster_imbalance, 0, cluster_imbalance
        print(f"当前集群Cost: {cluster_cost:.2f}，开始采样计算")
        for z in Z:
            mark_time = time()

            cost = 0
            bal, mig, value = 0, 0, 0
            machine_cpu_datas_at_t, machine_mem_datas_at_t = None, None

            # 我猜测 bal 是负载不均衡度，mig 是迁移成本，value 是总成本
            # 但是我不理解 bal|balx|balf、mig|migx|migf 和 value|valuex|valuef 是什么

            # 滑动窗口，每次移动一个时间片
            for t in range(W):
                if t == 1 and timeslot + W - 1 >= length_of_cpu_list:
                    break
                candidate = {}
                
                if t != 0 and is_satisfies_constraints:
                    imbalanced_machines = self.server_classification(cluster, b, y, t)
                is_satisfies_constraints = False

                # K次尝试
                for _ in range(K):
                    machine_cpu_datas_at_t, machine_mem_datas_at_t, is_satisfies_constraints = self.random_greedy_schedule_containers(M, a, b, v, t, imbalanced_machines, candidate, machine_cpu_datas_at_t, machine_mem_datas_at_t)
                    # 本次尝试满足约束，结束后面更多的尝试
                    if is_satisfies_constraints:
                        break
                    candidate.clear()
                
                if t == 0:
                    z_2_candidate[z] = candidate

                if len(candidate) == 0 or is_satisfies_constraints == False:
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

            print("[采样轮次Z=%d] 本轮Cost计算消耗 %.2fs，当前Cost: %.3f" % (z, time()-mark_time, cost))
            if cost < cluster_cost:
                # 更新 cluster_cost
                cluster_cost = cost
                min_z_record, imbalance_cost, migrate_cost, all_cost = z, bal, mig, value
                print("[采样轮次Z=%d] 更新Cost，当前最优Cost: %.3f" % (z, cluster_cost))

            # 计算Cost并不耗时，更耗时的在这一步；原函数名是 backZero，个人理解是恢复集群到初始状态，进行下一轮采样
            cluster.recover_cluster(z, timeslot, W)

        # 基于计算结果进行Instance的调度，我理解模拟环境下只需更新字段即可，不需要真正迁移
        motivation = cluster.remap_instance_to_machine(z_2_candidate, min_z_record, timeslot, W, b)
        print(f"Motivation: {motivation}")
        if len(motivation) > 0:
            motivation_file = self.motivation_file
            
            with open(motivation_file, "a") as f:
                writer = csv.writer(f)
                
                for containerId, metric in motivation.items():
                    writer.writerow([timeslot, containerId, metric])

        return min_z_record, imbalance_cost, migrate_cost, all_cost

    """
    原函数名是 findOverAndUnder，我理解对应Algo.1中的 server classification 阶段
    找到一个上界和下界，进而可以划分出2个Node集合：待迁出Node集合和待迁入Node集合
        :param cluster: Cluster对象
        :param b:  Beta，β，内存资源相比于CPU资源的归一化参数，见公式(1)
        :param y:  归一化参数，暂时不了解作用，未在论文中体现
        :param timeslot: 时间片
        
        :returns: migrate_out_machine_ids, migrate_in_machine_ids
        :returns: 超过阈值的MachineID列表（待迁出），低于阈值的MachineID列表（待迁入）
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
        migrate_out_machine_ids = machine_ids[load_over_threshold]
        # 负载不均衡度小于迁入阈值的Machine，即待迁入Machine
        load_under_threshold = np.where(load_imbalance_degree < threshold_dest)[0]
        migrate_in_machine_ids = machine_ids[load_under_threshold]
        
        return migrate_out_machine_ids, migrate_in_machine_ids

    # (1) 从待迁出Node集合中通过LHS随机采样得到一批Pod
    # (2) 计算各个Pod迁移到待迁入Node的Cost，取Cost最小的迁移方案
    # (3) 如果所有Pod的调度结果都满足Node剩余资源的约束，则返回True，否则返回False；返回False会进行下一轮抽样尝试，最多由外层尝试K次
    def random_greedy_schedule_containers(self, M, a, b, v, t, imbalanced_machines, candidate, machine_cpu_datas_at_t=None, machine_mem_datas_at_t=None):
        cluster = self.cluster
        machines = cluster.machines
        sum_of_cpu = cluster.sum_of_cpu
        sum_of_mem = cluster.sum_of_mem

        instance_cpu_datas_at_t = list(cluster.instance_cpu[:, t])
        instance_mem_datas_at_t = list(cluster.instance_mem[:, t])

        migrate_out_machine_ids, migrate_in_machine_ids = imbalanced_machines

        # 感觉代码的写法上有点问题，machine_xxx_datas_at_t 是 t 时刻的资源数据，下面尝试调度后会更新
        # 但可能调度几个Instance后不满足Algo.1中的约束(16行)，要直接舍弃 machine_xxx_datas_at_t，所以返回None
        #   如果返回None，下次再尝试会重新计算 machine_xxx_datas_at_t，不受本次影响
        #   如果不返回None，machine_xxx_datas_at_t 会在下次调用当前函数时作为参数传入，继续累积更新
        if machine_cpu_datas_at_t is None or machine_mem_datas_at_t is None:
            machine_cpu_datas_at_t = {k: sum_of_cpu_list[t] for k, sum_of_cpu_list in sorted(sum_of_cpu.items(), key=lambda x: x[0])}
            machine_mem_datas_at_t = {k: sum_of_mem_list[t] for k, sum_of_mem_list in sorted(sum_of_mem.items(), key=lambda x: x[0])}
            machine_cpu_datas_at_t = np.array(list(machine_cpu_datas_at_t.values()))
            machine_mem_datas_at_t = np.array(list(machine_mem_datas_at_t.values()))

        # 从待迁出Machine上通过LSH选择一批Instance进行迁移
        for machine_id in migrate_out_machine_ids:
            this_machine = machines[machine_id]

            all_instances_ids = np.array([x for x in this_machine.instances.keys()])

            samples = int(np.ceil(v * len(all_instances_ids)))
            # lhs_data = lhs(1, samples)
            # 测试用，固定每次整体运行的随机数
            # 注意不是每次调用该函数的随机数，所以sampler是作为参数传入的；换言之，需要在timeslot迭代之前生成sampler
            lhs_data = self.sampler.random(samples)

            ids_index = lhs_data * len(all_instances_ids)
            ids_index = ids_index[:, 0].astype(int)

            migrate_instance_ids = np.unique(all_instances_ids[ids_index]).astype(int)
            
            for migrate_instance_id in migrate_instance_ids:
                migrate_instance_id = int(migrate_instance_id)

                # Instance CPU * ?
                bal_d_cpu = instance_cpu_datas_at_t[migrate_instance_id] * (machine_cpu_datas_at_t[machine_id] - instance_cpu_datas_at_t[migrate_instance_id] - machine_cpu_datas_at_t[migrate_in_machine_ids])
                bal_d_mem = instance_mem_datas_at_t[migrate_instance_id] * (machine_mem_datas_at_t[machine_id] - instance_mem_datas_at_t[migrate_instance_id] - machine_mem_datas_at_t[migrate_in_machine_ids])

                # 应该对应公式(1)，负载不均衡度由CPU和内存资源共同决定，并且设置归一化参数 b
                imbalance_cost = np.array(bal_d_cpu + b * bal_d_mem)
                # 对应公式(3)，迁移代价取决于内存资源
                migrate_cost = np.array(a * (M-1) * instance_mem_datas_at_t[migrate_instance_id])
                # 选择负载不均衡度大于迁移代价的Machine，这点在论文中有体现吗？
                idx = np.array(np.where(imbalance_cost > migrate_cost)[0])

                lendx = len(idx)
                if lendx == 0:
                    continue
                allmetric = imbalance_cost

                tmps = {migrate_in_machine_ids[idx[i]]: allmetric[idx[i]] for i in range(lendx)}
                candiUnder = [k for k, v in sorted(
                    tmps.items(), key=lambda x:x[1], reverse=True)]

                for destination in candiUnder:
                    cpu_if_migrate_in = machine_cpu_datas_at_t[destination] + instance_cpu_datas_at_t[migrate_instance_id]
                    mem_if_migrate_in = machine_mem_datas_at_t[destination] + instance_mem_datas_at_t[migrate_instance_id]

                    # 满足资源容量约束
                    if destination != machine_id and cpu_if_migrate_in < this_machine.cpu_capacity and mem_if_migrate_in < this_machine.mem_capacity:
                        machine_cpu_datas_at_t[machine_id] -= instance_cpu_datas_at_t[migrate_instance_id]
                        machine_cpu_datas_at_t[destination] = cpu_if_migrate_in
                        machine_mem_datas_at_t[machine_id] -= instance_mem_datas_at_t[migrate_instance_id]
                        machine_mem_datas_at_t[destination] = mem_if_migrate_in
                        
                        if migrate_instance_id not in candidate:
                            candidate[migrate_instance_id] = [(machine_id, destination)]
                        else:
                            candidate[migrate_instance_id].append((machine_id, destination))
                        break

        if len(candidate) > 0:
            for k in machines.keys():
                if machines[k].cpu_capacity < machine_cpu_datas_at_t[k] or machines[k].mem_capacity < machine_mem_datas_at_t[k]:
                    return None, None, False
        
        if len(candidate) <= 0:
            return None, None, False
        
        return machine_cpu_datas_at_t, machine_mem_datas_at_t, True
