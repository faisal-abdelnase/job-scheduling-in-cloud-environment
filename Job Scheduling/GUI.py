from flask import Flask, render_template, request, jsonify
from ACO_GA import ACO_GA_Scheduler
from antColonySystem import ACO_Scheduler
from antSystem import AS_Scheduler
from elitistAntSystem import EAS_Scheduler
from maxMinAntSystem import MMAS_Scheduler
from rankBasedAntSystem import ASrank_Scheduler
from task import Task
from virtualMachine import CloudNode

app = Flask(__name__)

tasks = []
nodes = []



@app.route('/')
def index():
    return render_template('index.html')




@app.route('/run/<algorithm>', methods=['POST'])


def run_algo(algorithm):
    global tasks, nodes
    
    data = request.get_json()
    # Read parameters from request
    num_tasks = data.get('tasks')
    num_nodes = data.get('nodes')
    num_ants = data.get('ants')
    max_iter = data.get('iterations')

    # Create fresh instances of tasks and cloud nodes
    # Create only if empty
    if not tasks:
        tasks = Task.create_tasks(num_tasks)
    if not nodes:
        nodes = CloudNode.create_cloud_nodes(num_nodes)

    

    # Select algorithm
    if algorithm == 'AS':
        scheduler = AS_Scheduler(tasks, nodes, num_ants=num_ants, max_iter=max_iter)
    elif algorithm == 'MMAS':
        scheduler = MMAS_Scheduler(tasks, nodes, num_ants=num_ants, max_iter=max_iter)
    elif algorithm == 'ASrank':
        scheduler = ASrank_Scheduler(tasks, nodes, num_ants=num_ants, max_iter=max_iter)

    elif algorithm == 'EAS':
        scheduler = EAS_Scheduler(tasks, nodes, num_ants=num_ants, max_iter=max_iter)

    elif algorithm == 'ACO':
        scheduler = ACO_Scheduler(tasks, nodes, num_ants=num_ants, max_iter=max_iter)

    elif algorithm == 'ACO_GA':
        scheduler = ACO_GA_Scheduler(tasks, nodes, num_ants=num_ants, max_iter=max_iter)
    else:
        return jsonify({'error': 'Invalid algorithm'}), 400

    # Run the scheduler
    logs, best_makespan, fitness_logs, convergence_history = scheduler.run()

    # Collect best node assignment data
    node_stats = [{
        'id': node.id,
        'cpu_capacity': node.cpu_capacity,
        'memory_capacity': node.memory_capacity,
        'task_count': len(node.assigned_tasks),
        'utilization': node.calculate_utilization() * 100,
        'total_duration': sum(t.duration for t in node.assigned_tasks)
    } for node in nodes]

    return jsonify({
        'logs': logs,
        'best_makespan': best_makespan,
        'node_utilization': node_stats,
        "fitness_logs": fitness_logs,
        "convergence_history": convergence_history
    })


if __name__ == '__main__':
    app.run(debug=True)
