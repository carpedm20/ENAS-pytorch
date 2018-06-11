import numpy as np
from models import cnn_layers

"""
Contains Helper methods for sampling and fixing dags.
"""

def sample(probs):
    """Samples each index with the given probablities."""
    random_nums = np.random.rand(len(probs))
    return np.squeeze(np.argwhere(probs >= random_nums))


def sample_dag(dag_probs, all_connections):
    """Samples a dag with the given probabilities."""
    dag_idx = sample(dag_probs)
    # reduction_dag_idx = sample(reduction_dag_probs)
    if dag_idx.size > 0:
        dag = list(all_connections[i] for i in np.nditer(dag_idx))
    else:
        dag = []

    dag.sort()
    return dag


def trim_dag_one_way(dag: list, start_nodes: set, forward=True):
    """
    Returns new Dag with nodes and edges removed from dag that are not accessible starting at the start_nodes.
    If forward is False, edge directions are reversed.

    :arg dag: List[(source_index, target_index)]
    """
    start_nodes = set(start_nodes)
    if forward:
        start_index, end_index = 0, 1
    else:
        start_index, end_index = 1, 0
    new_dag = []
    while start_nodes:
        current = start_nodes.pop()
        connections = list(conn for conn in dag if conn[start_index] == current)
        for conn in connections:
            start_nodes.add(conn[end_index])
        new_dag.extend(connections)
    return new_dag


def trim_dag(dag: list, start_nodes: set, output_node=None):
    """
    Returns new Dag with nodes and edges removed from dag that are not accessible starting from the start_nodes
    or cannot reach the output_node.
    If forward is False, edge directions are reversed.

    :arg dag: List[(source_index, target_index)]
    """
    new_dag = trim_dag_one_way(dag, start_nodes, True)

    if output_node and not any(conn[1] == output_node for conn in new_dag):
        raise RuntimeError("Invalid dag")

    if output_node:
        new_dag = trim_dag_one_way(new_dag, {output_node}, False)

    new_dag.sort()
    return new_dag


def sample_valid_dag(dag_probs, reduction_dag_probs, all_connections, num_blocks):
    """
    Samples dags with the given probabilities until valid dags are sampled.
    Valid dags are defined as having a valid path from node 0 or 1 to the last node with index (1+num_blocks).
    """
    while True:
        dag, reduction_dag = sample_dag(dag_probs, all_connections), sample_dag(reduction_dag_probs, all_connections)
        try:
            dag = trim_dag(dag, {0, 1}, 1 + num_blocks)
            reduction_dag = trim_dag(reduction_dag, {0, 1}, 1 + num_blocks)
            return dag, reduction_dag
        except RuntimeError:
            print(f"Invalid Dags {dag}, {reduction_dag}")
            print(f"probs: {dag_probs}, {reduction_dag_probs}")
            continue


def fix_dag_one_way(dag: list, start_nodes: set, forward=True):
    """
    Returns new Dag with edges added so that all edges have a valid path from starting a start_nodes.
    If a node is inaccessible, an identiy edge is added from the last node which has a valid path from a start node to
    the inaccessible node.

    :arg dag: List[(source_index, target_index)]
    """
    new_dag = list(dag)
    start_nodes = set(start_nodes)
    if forward:
        start_index, end_index = 0, 1
    else:
        start_index, end_index = 1, 0

    dag = list(dag)
    dag.sort(key=lambda x: (x[start_index], x[end_index]))
    if not forward:
        dag.reverse()

    for connection in dag:
        start = connection[start_index]
        end = connection[end_index]

        if start not in start_nodes:
            if forward:
                begin = max(node for node in start_nodes if node < start)
            else:
                begin = min(node for node in start_nodes if node > start)
            new_conn = [None, None, None]
            new_conn[start_index] = begin
            new_conn[end_index] = start
            new_conn[2] = cnn_layers.identity.__name__
            new_dag.append(tuple(new_conn))
            start_nodes.add(start)

        start_nodes.add(end)

    new_dag.sort()
    return new_dag


def fix_dag(dag: list, start_nodes: set, output_node=None):
    """
        Returns new Dag with edges added so that all edges have a valid path from starting a start_nodes and a
        valid path to the output node.

        :arg dag: List[(source_index, target_index)]
    """
    dag.sort()
    new_dag = fix_dag_one_way(dag, start_nodes, True)

    if output_node:
        new_dag = fix_dag_one_way(new_dag, {output_node}, False)

    new_dag.sort()

    # Check if new_dag is empty
    if not new_dag:
        return [(max(start_nodes), output_node, cnn_layers.identity.__name__)]

    return new_dag


def sample_fixed_dag(dag_probs, all_connections, num_blocks):
    """
        Samples a dag with the given probabilities add edges so that they are valid.
        Valid dags are defined as having a valid path from node 0 or 1 to the last node with index (1+num_blocks).
    """
    dag = sample_dag(dag_probs, all_connections)
    dag = fix_dag(dag, {0, 1}, 1 + num_blocks)
    return dag


def sample_fixed_dags(dags_probs, all_connections, num_blocks):
    """
    Samples a dag with the given probabilities add edges so that they are valid.
    Valid dags are defined as having a valid path from node 0 or 1 to the last node with index (1+num_blocks).
    """
    return [sample_fixed_dag(dag_probs, all_connections=all_connections, num_blocks=num_blocks) for dag_probs in
            dags_probs]


def get_random(all_connections, num_blocks):
    """
    Returns a random trimmed dag where each layer has two random connections to it.
    """
    random_dag = []
    # random_dag.append((0, 1, np.random.choice(shared_cnn.CNNCell.default_layer_types)))
    for i in range(2, 2 + num_blocks):
        # previous_layers = np.random.choice(possible_connections, 2, replace=True)
        possible_connections = list(connection for connection in all_connections if connection[1] == i)
        # print(possible_connections)
        ids = np.random.choice(len(possible_connections), 2, replace=True)
        for id in ids:
            random_dag.append(possible_connections[id])

    return trim_dag(random_dag, {0, 1}, 1 + num_blocks)
