
import copy
import numpy as np
import matplotlib.pyplot as plt


class MMAS_Scheduler:
    def __init__(self, tasks, nodes, num_ants=10, max_iter=50, 
                    alpha=1.0, beta=2.0, rho=0.1, pheromone_min=0.1, 
                    pheromone_max=2.0, best_ants=2, q=1.0):
        self.tasks = tasks
        self.nodes = nodes
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance
        self.rho = rho      # Evaporation rate
        self.q = q
        
        self.pheromone_min = pheromone_min
        self.pheromone_max = pheromone_max
        self.best_ants = best_ants  # Number of best ants to update trails
        
        # Initialize pheromone matrix to max value
        self.pheromone = np.ones((len(tasks), len(nodes))) * pheromone_max
        
        
        self.convergence_history = []
    


    # Calculate heuristic value for task-node pair
    def calculate_heuristic(self, task, node):
        
        # Combine execution time and load balance factors
        exec_time = task.duration / (node.available_cpu + 0.001)
        load_balance = 1 / (node.calculate_utilization() + 0.001)
        return 1 / (exec_time * (1 + load_balance))
    

    # Calculate probability of assigning task to node
    def calculate_probability(self, task_idx, node_idx):
        
        pheromone = self.pheromone[task_idx][node_idx] ** self.alpha
        heuristic = self.calculate_heuristic(self.tasks[task_idx], self.nodes[node_idx]) ** self.beta
        return pheromone * heuristic
    

    # Assign tasks to nodes for one ant
    def assign_tasks(self, ant):
        
        # Reset nodes
        for node in self.nodes:
            node.available_cpu = node.cpu_capacity
            node.available_memory = node.memory_capacity
            node.assigned_tasks = []
        
        # Sort tasks by resource requirements (largest first)
        sorted_tasks = sorted(self.tasks, key=lambda x: (x.cpu_requirement, x.memory_requirement), reverse=True)
        wait_list = []
        
        for task in sorted_tasks:
            probabilities = []
            valid_nodes = []
            
            # Calculate probabilities for valid nodes
            for node_idx, node in enumerate(self.nodes):
                if (node.available_cpu >= task.cpu_requirement and 
                    node.available_memory >= task.memory_requirement):
                    prob = self.calculate_probability(self.tasks.index(task), node_idx)
                    probabilities.append(prob)
                    valid_nodes.append(node_idx)
            
            if not valid_nodes:
                wait_list.append(task)
                continue
            
            # Normalize probabilities
            total = sum(probabilities)
            if total == 0:
                probabilities = [1/len(valid_nodes)] * len(valid_nodes)
            else:
                probabilities = [p/total for p in probabilities]
            
            # Select node based on probabilities
            selected_idx = np.random.choice(valid_nodes, p=probabilities)
            selected_node = self.nodes[selected_idx]
            
            # Assign task
            selected_node.assign_task(task)
            
            # Apply local pheromone update (different from standard ACO)
            current_pheromone = self.pheromone[self.tasks.index(task)][selected_idx]
            self.pheromone[self.tasks.index(task)][selected_idx] = max(
                self.pheromone_min,
                (1 - self.rho) * current_pheromone + self.rho * self.pheromone_min
            )

        return wait_list
    

    # Calculate makespan (maximum completion time across nodes)
    def calculate_makespan(self):
        
        makespan = 0
        for node in self.nodes:
            node_makespan = sum(task.duration for task in node.assigned_tasks)
            if node_makespan > makespan:
                makespan = node_makespan
        return makespan
    

    # Calculate fitness of solution (higher is better)
    def calculate_fitness(self, makespan):
        
        # Incorporate load balancing
        utilizations = [node.calculate_utilization() for node in self.nodes]
        load_balance = 1 / (np.std(utilizations) + 0.001)
        
        # Incorporate wait time (simple approximation)
        avg_wait = sum(sum(t.duration for t in node.assigned_tasks[:-1]) for node in self.nodes) / len(self.tasks)
        
        return (1 / (makespan + 0.001)) * load_balance * (1 / (avg_wait + 0.001))
    

    # Update pheromone trails based on best solutions only
    def update_pheromone(self, solutions):
        
        # Sort solutions by fitness (descending)
        sorted_solutions = sorted(solutions, key=lambda x: x['fitness'], reverse=True)
        
        # Evaporate all pheromone trails
        self.pheromone *= (1 - self.rho)
        
        # Only allow best ants to deposit pheromone
        for solution in sorted_solutions[:self.best_ants]:
            fitness = solution['fitness']
            for task_idx, node_idx in solution['assignments']:
                # Calculate new pheromone with bounds enforcement
                new_pheromone = self.pheromone[task_idx][node_idx] + (self.q * fitness)
                self.pheromone[task_idx][node_idx] = min(
                    self.pheromone_max,
                    max(self.pheromone_min, new_pheromone)
                )
        
        # Ensure all pheromones stay within bounds
        self.pheromone = np.clip(self.pheromone, self.pheromone_min, self.pheromone_max)
    

    
    # Measure how concentrated the pheromone trails are
    def calculate_convergence(self):
        
        return np.std(self.pheromone) / np.mean(self.pheromone)

    # Run the MMAS algorithm
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
            
            # Let each ant find a solution
            for ant in range(self.num_ants):
                wait_list = self.assign_tasks(ant)
                makespan = self.calculate_makespan()
                fitness = self.calculate_fitness(makespan)
                
                # Record assignments
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
                    best_solution = copy.deepcopy(solution)
                
            
            # Update pheromone trails based on best solutions
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
    

    