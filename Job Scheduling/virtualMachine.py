
import random


class CloudNode:
    def __init__(self, node_id, cpu_capacity, memory_capacity):
        self.id = node_id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.available_cpu = cpu_capacity
        self.available_memory = memory_capacity
        self.assigned_tasks = []
        self.utilization_history = []
    
    def assign_task(self, task):
        if (self.available_cpu >= task.cpu_requirement and 
            self.available_memory >= task.memory_requirement):
            self.available_cpu -= task.cpu_requirement
            self.available_memory -= task.memory_requirement
            self.assigned_tasks.append(task)
            task.assigned_node = self.id
            return True
        return False
    
    def calculate_utilization(self):
        cpu_util = 1 - (self.available_cpu / self.cpu_capacity)
        mem_util = 1 - (self.available_memory / self.memory_capacity)
        return (cpu_util + mem_util) / 2  # Average utilization
    

    # Create a list of cloud nodes with varying capacities
    def create_cloud_nodes(num_nodes=10):
        nodes = []
        for i in range(num_nodes):
            # Vary capacities between 8-32 CPU cores and 16-64GB memory
            cpu = random.choice([8, 16, 24, 32])
            memory = cpu * random.choice([2, 3, 4])
            nodes.append(CloudNode(f"Node-{i+1}", cpu, memory))
        return nodes