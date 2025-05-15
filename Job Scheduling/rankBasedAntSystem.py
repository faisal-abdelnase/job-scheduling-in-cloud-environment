import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime

class ASrank_Scheduler:
    def __init__(self, tasks, nodes, num_ants=10, max_iter=50, 
                    alpha=1.0, beta=2.0, rho=0.1, q=1.0, elitist_weight=2, w = 5):
        self.tasks = tasks
        self.nodes = nodes
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha    # Pheromone importance
        self.beta = beta      # Heuristic importance
        self.rho = rho        # Evaporation rate
        self.q = q            # Pheromone deposit factor
        self.w = w
        self.elitist_weight = elitist_weight  # Weight for best solutions
        
        # Initialize pheromone matrix
        self.pheromone = np.ones((len(tasks), len(nodes))) * 0.1
        

        self.convergence_history = []
    

    # Calculate heuristic desirability of assigning task to node
    def calculate_heuristic(self, task, node):
        
        # Consider both execution time and load balancing
        exec_time = task.duration / (node.available_cpu + 0.001)
        load_balance = 1 / (node.calculate_utilization() + 0.001)
        return 1 / (exec_time * (1 + load_balance))
    


    # Calculate probability of assigning task to node
    def calculate_probability(self, task_idx, node_idx):
        
        pheromone = self.pheromone[task_idx][node_idx] ** self.alpha
        heuristic = self.calculate_heuristic(self.tasks[task_idx], self.nodes[node_idx]) ** self.beta
        return pheromone * heuristic
    

    # Assign tasks to nodes for one ant
    def assign_tasks(self):
        
        # Reset nodes
        for node in self.nodes:
            node.available_cpu = node.cpu_capacity
            node.available_memory = node.memory_capacity
            node.assigned_tasks = []
        
        # Sort tasks by size (largest first) for better load balancing
        sorted_tasks = sorted(self.tasks, key=lambda x: (x.cpu_requirement, x.memory_requirement), reverse=True)
        wait_list = []
        
        for task in sorted_tasks:
            valid_nodes = []
            probabilities = []
            
            # Calculate probabilities for valid nodes
            for node_idx, node in enumerate(self.nodes):
                if (node.available_cpu >= task.cpu_requirement and 
                    node.available_memory >= task.memory_requirement):
                    prob = self.calculate_probability(self.tasks.index(task), node_idx)
                    probabilities.append(prob)
                    valid_nodes.append(node_idx)
            
            if not valid_nodes:
                wait_list.append(task)
                continue  # Skip if no valid node found
            
            # Normalize probabilities
            total = sum(probabilities)
            if total == 0:
                probabilities = [1/len(valid_nodes)] * len(valid_nodes)
            else:
                probabilities = [p/total for p in probabilities]
            
            # Select node based on probabilities
            selected_idx = np.random.choice(valid_nodes, p=probabilities)
            selected_node = self.nodes[selected_idx]
            selected_node.assign_task(task)

        return wait_list
    

    # Calculate makespan (maximum completion time across nodes)
    def calculate_makespan(self):
        
        return max(sum(task.duration for task in node.assigned_tasks) for node in self.nodes)
    

    # Calculate fitness of solution (higher is better)
    def calculate_fitness(self, makespan):
        
        # Incorporate load balancing
        utilizations = [node.calculate_utilization() for node in self.nodes]
        load_balance = 1 / (np.std(utilizations) + 0.001)
        
        # Incorporate wait time (simple approximation)
        avg_wait = sum(sum(t.duration for t in node.assigned_tasks[:-1]) for node in self.nodes) / len(self.tasks)
        
        return (1 / (makespan + 0.001)) * load_balance * (1 / (avg_wait + 0.001))
    
    

    # Update pheromone trails using rank-based approach
    def update_pheromone(self, ranked_solutions):
        
        # Evaporate pheromone
        self.pheromone *= (1 - self.rho)
        
        # Update pheromone based on ranked solutions
        for rank, solution in enumerate(ranked_solutions, 1):
            assignments = solution['assignments']
            fitness = solution['fitness']
            weight = (self.num_ants - rank) * self.q
            if rank == 1:  # Additional weight for best solution
                weight *= self.elitist_weight
                
            for task_idx, node_idx in assignments:
                self.pheromone[task_idx][node_idx] += weight * fitness
        


    # Measure how concentrated the pheromone trails are
    def calculate_convergence(self):
        
        return np.std(self.pheromone) / np.mean(self.pheromone)

    # Run the ASrank algorithm
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
            
            # Generate solutions from all ants
            for _ in range(self.num_ants):
                wait_list = self.assign_tasks()
                makespan = self.calculate_makespan()
                fitness = self.calculate_fitness(makespan)
                
                # Record the solution
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
                
                # Track best solution
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = deepcopy(solution)
            
            # Rank solutions by makespan (ascending)
            ranked_solutions = sorted(solutions, key=lambda x: x['makespan'])
            
            # Update pheromone trails
            self.update_pheromone(ranked_solutions)

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
    

    