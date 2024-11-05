import os
import numpy as np
import pandas as pd
from instance import InstanceConfig
from machine import MachineConfig

"""
加载数据集，以文件名作为ID，记录对应的CPU和内存请求曲线
"""
def read_all_files(filepath, max_files=None):
    cpu_list = {}
    mem_list = {}
    files = os.listdir(filepath)

    # Debug的时候没有必要每次都加载所有数据集，太慢
    if max_files is not None:
        files = files[:max_files]

    for idx, file in enumerate(files):
        filename = os.path.join(filepath, file)
        ids = int(filename[filename.rfind("_")+1 : filename.rfind(".")])
        
        df = pd.read_csv(filename, header=None)

        cpu = df[0].values.tolist()
        mem = df[1].values.tolist()
        cpu_list[ids] = cpu
        mem_list[ids] = mem
    return cpu_list, mem_list

"""
加载数据集
    我理解加载数据集，创建MachineConfig和InstanceConfig对象，不应该这么复杂，最多是重新映射两者的ID
    :param instance_cpu_and_mem_files: 数据集存放路径
    :param test_array: 测试集，结构为：[[节点数，容器数], ...]
"""
def load_instance_data(instance_cpu_and_mem_files, test_array):

    # 为什么原代码将变量命名为vm_cpu_requests、vm_mem_requests？vm是什么意思？
    instance_cpu_requests, instance_mem_requests = read_all_files(instance_cpu_and_mem_files)

    # 读取第一时刻(最初)Instance-Machine的放置关系
    instance_machine_id_filepath = "data/container_machine_id.csv" # 容器与机器对应关系的文件路径
    df_instance_2_machine = pd.read_csv(instance_machine_id_filepath, header=None)

    # {新ID: 旧ID}的映射关系，新ID是连续的、旧ID是不连续的
    instance_ids = {}
    # 记录Machine运行有哪些Container，{machine_id: [instance_id, ...]}
    machine = {}
    # container_machine_id.csv记录的是Instance向Machine的映射，但InstanceID并不是连续的
    # 修改为连续的InstanceID；用 instance_ids 记录{新ID: 旧ID}的映射
    for idx, instance_2_machine in df_instance_2_machine.iterrows():
        instance_ids[idx] = instance_2_machine[0]
        machine_id = instance_2_machine[1]
        
        if machine_id in machine:
            machine[machine_id].append(idx)
        else:
            machine[machine_id] = [idx]
    # 按MachineID进行排序
    machine = dict(sorted(machine.items()))

    # MachineID也不是连续的，也需要重新构建连续的ID；重建后machine变成machine_new
    machine_ids = {}
    machine_new = {}

    idx = 0
    # 将Machine重新整合到 machine_new 中
    for key, value in machine.items():
        machine_new[idx] = value
        machine_ids[idx] = key # 存储Machine的{新ID: 旧ID}
        idx = idx + 1

    # 原本有一个machine_half变量，用于存储一半的机器配置，但实际发现根本没有用途，于是删除

    machine_key_filepath = "data/machine_keys/3989.csv"
    machine_keys = pd.read_csv(machine_key_filepath)
    
    result = []
    instance_id_2_instance_config = {} # 存储实例配置信息 {instance_id : InstanceConfig}
    machine_id_2_machine_config = {} # 存储机器配置信息 {machine_id : MachineConfig}
    # 根据test_array中的每个元组，生成Machine和Instance的配置信息
    for test_case in test_array:
        # 原本是叫nodeNum和containerNum，能不能统一一下术语，又是Instance又是Container，又是Node又是Machine的；还有命名风格
        # 【?】instance_num没有用到？
        machine_num, instance_num = test_case

        # 读取前node_num个MachineID
        some_machines = machine_keys["machine_id"].values.tolist()[:machine_num]
        machine = {key : machine_new[key] for key in some_machines}
        print(f"Len of machine is {len(machine)}")
        
        for machine_id, machine_instance_ids in machine.items():
            machine_config = MachineConfig(machine_id, 30, 100) # CPU和内存容量固定是30和100？
            machine_id_2_machine_config[machine_id] = machine_config
            
            for instance_id in machine_instance_ids:
                # 根据旧容器ID，获取对应的CPU请求曲线、内存请求曲线
                cpu_curve = instance_cpu_requests[instance_ids[instance_id]]
                memory_curve = instance_mem_requests[instance_ids[instance_id]]
                disk_curve = np.zeros_like(cpu_curve)   # disk_curve的意义是？本身就不存在对应的数据
                instance_config = InstanceConfig(machine_id, instance_id, cpu_curve[0], memory_curve[0], disk_curve, cpu_curve, memory_curve)
                instance_id_2_instance_config[instance_id] = instance_config
        print(f'Len of instance_configs is {len(instance_id_2_instance_config)}')

        # 重新分配Machine和Instance的ID，前面还没涉及到Config对象
        i = 0
        j = 0
        new_machines  = {}
        new_instances = {}
        
        for machine_id, instance_list in machine.items():
            machine_config = machine_id_2_machine_config[machine_id]
            machine_config.id = j
            new_machines[j] = machine_config
            j += 1
            
            for instance_id in instance_list:
                instance_config = instance_id_2_instance_config[instance_id]
                instance_config.machine_id = machine_config.id
                instance_config.id = i
                new_instances[i] = instance_config
                i += 1
        result.append([new_instances, new_machines, machine_ids, instance_ids])

    print(f"Len of result is {len(result)}")
    return result
