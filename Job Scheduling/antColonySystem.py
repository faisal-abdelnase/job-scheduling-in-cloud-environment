
import random
import math
import copy
import numpy as np


class ACO_Scheduler:
    def __init__(self, tasks, nodes, num_ants=10, max_iter=50, alpha=1.0, beta=2.0, rho=0.1, q=1.0):
        self.tasks = tasks
        self.nodes = nodes
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q

        self.pheromone = np.ones((len(tasks), len(nodes))) / len(nodes)

        self.convergence_history = []

    def calculate_heuristic(self, task, node):
        exec_time = task.duration / (node.available_cpu + 0.001)
        load_balance = 1 / (node.calculate_utilization() + 0.001)
        return 1 / (exec_time * load_balance)

    def calculate_probability(self, task_idx, node_idx):
        pheromone = self.pheromone[task_idx][node_idx] ** self.alpha
        heuristic = self.calculate_heuristic(self.tasks[task_idx], self.nodes[node_idx]) ** self.beta
        return pheromone * heuristic

    def assign_tasks(self, ant):
        for node in self.nodes:
            node.available_cpu = node.cpu_capacity
            node.available_memory = node.memory_capacity
            node.assigned_tasks = []

        shuffled_tasks = self.tasks[:]
        random.shuffle(shuffled_tasks)

        wait_list = []

        for task in shuffled_tasks:
            probabilities = []
            valid_nodes = []

            for node_idx, node in enumerate(self.nodes):
                if (node.available_cpu >= task.cpu_requirement and
                        node.available_memory >= task.memory_requirement):
                    prob = self.calculate_probability(self.tasks.index(task), node_idx)
                    probabilities.append(prob)
                    valid_nodes.append(node_idx)

            if not valid_nodes:
                wait_list.append(task)
                continue

            total = sum(probabilities)
            if total == 0:
                probabilities = [1 / len(valid_nodes)] * len(valid_nodes)
            else:
                probabilities = [p / total for p in probabilities]

            selected_idx = np.random.choice(valid_nodes, p=probabilities)
            selected_node = self.nodes[selected_idx]
            selected_node.assign_task(task)

            self.pheromone[self.tasks.index(task)][selected_idx] *= (1 - self.rho)
            self.pheromone[self.tasks.index(task)][selected_idx] += (self.rho * 0.1)

        return wait_list

    def calculate_makespan(self):
        makespan = 0
        for node in self.nodes:
            node_makespan = sum(task.duration for task in node.assigned_tasks)
            makespan = max(makespan, node_makespan)
        return makespan

    def calculate_fitness(self, makespan):
        utilizations = [node.calculate_utilization() for node in self.nodes]
        load_balance = 1 / (np.std(utilizations) + 0.001)
        avg_wait = sum(sum(t.duration for t in node.assigned_tasks[:-1]) for node in self.nodes) / len(self.tasks)
        return (1 / (makespan + 0.001)) * load_balance * (1 / (avg_wait + 0.001))

    def update_pheromone(self, solutions):
        self.pheromone *= (1 - self.rho)
        for solution in solutions:
            fitness = solution['fitness']
            for task_idx, node_idx in solution['assignments']:
                self.pheromone[task_idx][node_idx] += self.q * fitness

    # Measure how concentrated the pheromone trails are
    def calculate_convergence(self):
        
        return np.std(self.pheromone) / np.mean(self.pheromone)

    def run(self):
        best_solution = None
        best_fitness = -1
        iteration_logs = []
        fitness_logs = []

        for iteration in range(self.max_iter):
            solutions = []
            assign_count_total = 0
            wait_count_total = 0
            total_wait_time = 0

            for ant in range(self.num_ants):
                wait_list = self.assign_tasks(ant)
                makespan = self.calculate_makespan()
                fitness = self.calculate_fitness(makespan)

                assignments = []
                assigned_tasks = 0
                for node_idx, node in enumerate(self.nodes):
                    for task in node.assigned_tasks:
                        task_idx = self.tasks.index(task)
                        assignments.append((task_idx, node_idx))
                        assigned_tasks += 1

            
                
                solution = {
                    'assignments': assignments,
                    'makespan': makespan,
                    'fitness': fitness
                }
                solutions.append(solution)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = copy.deepcopy(solution)

            self.update_pheromone(solutions)


            wait_count = len(wait_list)

            avg_wait = sum(sum(t.duration for t in node.assigned_tasks[:-1]) for node in self.nodes) / len(self.tasks)
            

            log_msg = (
                f"Iteration {iteration + 1}: Best Makespan = {best_solution['makespan']}, "
                f"Assigned Tasks = {assigned_tasks}, Waiting Tasks = {wait_count}, "
                f"Avg Wait Time = {avg_wait:.2f}"
            )
            fitness_logs.append(best_solution['fitness'])
            iteration_logs.append(log_msg)
            self.convergence_history.append(self.calculate_convergence())
            print(log_msg)

        return iteration_logs, best_solution['makespan'], fitness_logs, self.convergence_history
