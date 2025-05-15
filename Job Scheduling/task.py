import random


class Task:
    def __init__(self, task_id, cpu_requirement, memory_requirement, duration):
        self.id = task_id
        self.cpu_requirement = cpu_requirement
        self.memory_requirement = memory_requirement
        self.duration = duration
        self.assigned_node = None
        self.start_time = None
        self.end_time = None

    def create_tasks(num_tasks=50):
        tasks = []
        for i in range(num_tasks):
            # Vary requirements - smaller tasks more common
            cpu_req = random.choices([1, 2, 4, 8], weights=[40, 30, 20, 10])[0]
            mem_req = cpu_req * random.choice([1, 2, 4])
            duration = random.randint(1, 10)  # Time units
            tasks.append(Task(f"Task-{i+1}", cpu_req, mem_req, duration))
        return tasks
    

