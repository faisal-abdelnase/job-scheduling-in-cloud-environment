import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class ACO_GA_Scheduler:
    def __init__(self, tasks, nodes, num_ants=10, max_iter=50, alpha=1.0, beta=2.0, rho=0.1, q=1.0,
                    ga_population_size=20, ga_generations=50, crossover_rate=0.8, mutation_rate=0.1):
        self.tasks = tasks
        self.nodes = nodes
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha  
        self.beta = beta    
        self.rho = rho      
        self.q = q          
        
        # GA parameters
        self.ga_population_size = ga_population_size
        self.ga_generations = ga_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        
        self.pheromone = np.ones((len(tasks), len(nodes))) / len(nodes)
        
        
        self.best_solutions_history = []
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
                continue
            
            total = sum(probabilities)
            if total == 0:
                probabilities = [1/len(valid_nodes)] * len(valid_nodes)
            else:
                probabilities = [p/total for p in probabilities]
            
            selected_idx = np.random.choice(valid_nodes, p=probabilities)
            selected_node = self.nodes[selected_idx]
            selected_node.assign_task(task)
            
            self.pheromone[self.tasks.index(task)][selected_idx] *= (1 - self.rho)
            self.pheromone[self.tasks.index(task)][selected_idx] += (self.rho * 0.1)

    def calculate_makespan(self):
        makespan = 0
        for node in self.nodes:
            node_makespan = sum(task.duration for task in node.assigned_tasks)
            if node_makespan > makespan:
                makespan = node_makespan
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

    
    def calculate_convergence(self):
        
        return np.std(self.pheromone) / np.mean(self.pheromone)

    
    def solution_to_chromosome(self, solution):
        
        chromosome = [-1] * len(self.tasks)
        for task_idx, node_idx in solution['assignments']:
            chromosome[task_idx] = node_idx
        return chromosome
    

    
    def chromosome_to_solution(self, chromosome):
        # Reset nodes
        for node in self.nodes:
            node.available_cpu = node.cpu_capacity
            node.available_memory = node.memory_capacity
            node.assigned_tasks = []
        
        
        for task_idx, node_idx in enumerate(chromosome):
            if node_idx == -1:
                continue  
            task = self.tasks[task_idx]
            node = self.nodes[node_idx]
            
            
            if (node.available_cpu >= task.cpu_requirement and 
                node.available_memory >= task.memory_requirement):
                node.assign_task(task)
            else:
                
                continue
        
    
        assignments = []
        for node_idx, node in enumerate(self.nodes):
            for task in node.assigned_tasks:
                task_idx = self.tasks.index(task)
                assignments.append((task_idx, node_idx))
        
        makespan = self.calculate_makespan()
        fitness = self.calculate_fitness(makespan)
        
        return {
            'assignments': assignments,
            'makespan': makespan,
            'fitness': fitness,
            'chromosome': chromosome.copy()
        }
    
    
    
    def initialize_ga_population(self, population_ACO):
        population = []
        
        
        for elite_group in population_ACO:
            for sol in elite_group:
                population.append(self.solution_to_chromosome(sol))
        
        
        while len(population) < self.ga_population_size:
            parent = random.choice(population)
            mutated = self.mutate_chromosome(parent.copy())
            population.append(mutated)
        
        return population
    


    
    def crossover(self, parent1, parent2):
        
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Single-point crossover
        point = random.randint(1, len(parent1)-2)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        return child1, child2


    
    def mutate_chromosome(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                # Mutate this gene (task assignment)
                valid_nodes = []
                task = self.tasks[i]
                for node_idx, node in enumerate(self.nodes):
                    if (node.cpu_capacity >= task.cpu_requirement and 
                        node.memory_capacity >= task.memory_requirement):
                        valid_nodes.append(node_idx)
                if valid_nodes:
                    chromosome[i] = random.choice(valid_nodes)
        return chromosome





    # Run the ACO algorithm
    def run_aco(self):
        
        best_fitness = -1
        population_ACO = []
        
        for iteration in range(self.max_iter):
            solutions = []
            
            
            for ant in range(self.num_ants):
                self.assign_tasks(ant)
                makespan = self.calculate_makespan()
                fitness = self.calculate_fitness(makespan)
                
                
                assignments = []
                for node_idx, node in enumerate(self.nodes):
                    for task in node.assigned_tasks:
                        task_idx = self.tasks.index(task)
                        assignments.append((task_idx, node_idx))
                
                solution = {
                    'assignments': assignments,
                    'makespan': makespan,
                    'fitness': fitness
                }
                solutions.append(solution)
                
            
            
            self.update_pheromone(solutions)
            solutions.sort(key=lambda x: x['fitness'], reverse=True)
            population_ACO.append(solutions[:5])
            self.convergence_history.append(self.calculate_convergence())
            
            
        
        return population_ACO



    # Run GA for several generations
    def run(self):
        population_ACO = self.run_aco()
        population = self.initialize_ga_population(population_ACO)
        iteration_logs = []
        fitness_logs = []
        
        for generations in range(self.ga_generations):
            
            evaluated = []
            for chromo in population:
                solution = self.chromosome_to_solution(chromo)
                evaluated.append((solution['fitness'], chromo))
            
            
            evaluated.sort(reverse=True, key=lambda x: x[0])
            
            
            elites = [chromo for (fit, chromo) in evaluated[:len(evaluated)//2]]
            
            
            new_population = elites.copy()
            
            while len(new_population) < self.ga_population_size:
                # Tournament selection
                candidates = random.sample(evaluated, min(4, len(evaluated)))
                candidates.sort(reverse=True, key=lambda x: x[0])
                parent1 = candidates[0][1]
                parent2 = candidates[1][1]
                
                # Crossover
                children = self.crossover(parent1, parent2)
                if children is None:
                    continue  
                
                child1, child2 = children
                
                # Mutate
                child1 = self.mutate_chromosome(child1)
                child2 = self.mutate_chromosome(child2)
                
                
                new_population.append(child1)
                if len(new_population) < self.ga_population_size:
                    new_population.append(child2)
            
            population = new_population
        
            
            best_solution = None
            best_fitness = -1
            for chromo in population:
                solution = self.chromosome_to_solution(chromo)
                if solution['fitness'] > best_fitness:
                    best_fitness = solution['fitness']
                    best_solution = solution

            log_msg = f"generation {generations + 1}: Best Makespan = {best_solution['makespan']}"
            fitness_logs.append(best_fitness)
            iteration_logs.append(log_msg)
        
            print(f"generation {generations+1}: Best Makespan = {best_solution['makespan']}")
        return iteration_logs, best_solution['makespan'], fitness_logs, self.convergence_history
        