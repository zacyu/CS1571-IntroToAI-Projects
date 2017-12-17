#!/usr/bin/env python3
"""
CS 1571 Project 1: Graph Search problems
Zac Yu (zhy46@)
"""

from argparse import ArgumentParser, FileType
from ast import literal_eval
from collections import deque, namedtuple
from datetime import datetime
from heapq import heapify, heappop, heappush
from math import inf, sqrt

DEBUG_MODE = False
REPORT_MODE = False
REPORT_TIMEOUT_SECS = 30 * 60
if REPORT_MODE:
    START_TIME = datetime.now()


class ProblemInterface(object):
    """Abstract class for a problem"""

    def __init__(self, data):
        self.data = data
        self.initial_state = self._build_initial_state(data)

    def _build_initial_state(self, data):
        """Build initial state from problem data"""
        raise NotImplementedError()

    def actions(self, state):
        """Get a list of all possible actions from a certain state"""

    def final_cost(self, node):
        """Return the final cost to reach a state"""
        raise NotImplementedError()

    def heuristic(self, node):
        """Estimate the cheapest cost to a goal state"""
        raise NotImplementedError()

    def goal_test(self, state):
        """Verify is a state is goal"""
        raise NotImplementedError()

    @classmethod
    def initial_path_cost(cls):
        """The cost of the initial path from origin"""
        return 0

    @classmethod
    def next_path_cost(cls, path_cost, step_cost):
        """Calculating the path cost with an additional step"""
        return path_cost + step_cost

    def result(self, state, action):
        """Transition from one state to another for some action"""
        raise NotImplementedError()

    def step_cost(self, state, action):
        """The cost to reach another state for some action"""
        raise NotImplementedError()


class Node(object):
    """Represents a node in a state space"""

    def __init__(self, problem, state, parent=None, action=None,
                 path_cost=0):
        self.state = state
        self.parent = parent
        self.path = parent.path + (state.name, ) if parent \
            is not None else (state.name, )
        self.action = action
        self.path_cost = path_cost if parent is not None \
            else problem.initial_path_cost()

    def __lt__(self, other):
        return 0

    def __repr__(self):
        return str({
            'state': self.state.name,
            'path': self.path
        })

    def child_node(self, problem, action):
        """State expansion function"""
        return Node(problem, problem.result(self.state, action),
                    self, action, problem.next_path_cost(
                        self.path_cost, problem.step_cost(self.state, action)))


State = namedtuple('State', ['name', 'data'])


class MonitorProblem(ProblemInterface):
    """Wireless sensor monitoring problem"""

    def _build_initial_state(self, data):
        return State('', {
            'monitored_target_ids': set(),
            'sensor_id': None,
            'target_id': None,
            'used_sensor_ids': set(),
        })

    def actions(self, state):
        feasible_actions = []

        unmonitored_target_count = len(self.data['targets']) - \
            len(state.data['monitored_target_ids'])
        used_sensor_count = len(state.data['used_sensor_ids'])
        unused_sensor_count = len(self.data['sensors']) - used_sensor_count

        if used_sensor_count == len(self.data['sensors']):
            return feasible_actions

        sensor = self.data['sensors'][used_sensor_count]
        for target in self.data['targets']:
            if (unmonitored_target_count < unused_sensor_count or
                    target[0] not in state.data['monitored_target_ids']):
                feasible_actions.append({
                    'sensor': sensor,
                    'target': target,
                })
        if unmonitored_target_count < unused_sensor_count:
            feasible_actions.append({
                'sensor': sensor,
                'target': None,
            })
        return feasible_actions

    def final_cost(self, node):
        """This is not actually a cost but the maximized profit"""
        return -node.path_cost

    @classmethod
    def get_monitoring_time(cls, sensor, target):
        """Calculate the duration the target can be monitored by the sensor"""
        distance = sqrt(pow(sensor[1] - target[1], 2) +
                        pow(sensor[2] - target[2], 2))
        return sensor[3] / distance

    def goal_test(self, state):
        for target in self.data['targets']:
            if target[0] not in state.data['monitored_target_ids']:
                return False
        return True

    def heuristic(self, node):
        min_duration = inf
        state = node.state
        unmonitored_targets = list(filter(
            lambda t: (t[0] not in state.data['monitored_target_ids']),
            self.data['targets']))
        unused_sensors = list(filter(
            lambda s: (s[0] not in state.data['used_sensor_ids']),
            self.data['sensors']))
        while unmonitored_targets:
            target = unmonitored_targets.pop()
            max_duration_sensor_id = 0
            for sensor_id in range(1, len(unused_sensors)):
                if self.get_monitoring_time(unused_sensors[sensor_id], target) \
                    > self.get_monitoring_time(
                        unused_sensors[max_duration_sensor_id], target):
                    max_duration_sensor_id = sensor_id
            min_duration = min(min_duration,
                               self.get_monitoring_time(
                                   unused_sensors[max_duration_sensor_id],
                                   target))
        return max(node.path_cost, -min_duration) - node.path_cost

    @classmethod
    def initial_path_cost(cls):
        return -inf

    @classmethod
    def next_path_cost(cls, path_cost, step_cost):
        return max(path_cost, step_cost)

    def result(self, state, action):
        monitored_target_ids = state.data['monitored_target_ids'].copy()
        if action['target'] is not None:
            monitored_target_ids.add(action['target'][0])
        used_sensor_ids = state.data['used_sensor_ids'] | \
            {action['sensor'][0]}
        state_data = {
            'monitored_target_ids': monitored_target_ids,
            'sensor_id': action['sensor'][0],
            'target_id': action['target'][0]
            if action['target'] is not None else None,
            'used_sensor_ids': used_sensor_ids,
        }
        state_name = (state_data['sensor_id'] + '_' +
                      state_data['target_id']) \
            if action['target'] is not None else ''
        return State(name=state_name, data=state_data)

    def step_cost(self, state, action):
        sensor = action['sensor']
        target = action['target']
        if target is None:
            return -inf
        return -self.get_monitoring_time(sensor, target)


class AggregationProblem(ProblemInterface):
    """Data aggregation problem"""

    def _build_initial_state(self, data):
        return State('', {
            'visited_node_ids': set(),
            'visited': set()
        })

    def actions(self, state):
        feasible_actions = []
        delay_dict = self.data['delay']
        node_ids = list(map(lambda n: n[0], self.data['nodes']))
        visited_node_ids = state.data['visited_node_ids']

        for node_id in node_ids:
            if (frozenset(visited_node_ids), node_id) not in \
                state.data['visited'] and (not state.name
                                           or delay_dict[frozenset([state.name, node_id])]
                                           is not None):
                feasible_actions.append(node_id)
        return feasible_actions

    def final_cost(self, node):
        return node.path_cost

    def goal_test(self, state):
        for node in self.data['nodes']:
            if node[0] not in state.data['visited_node_ids']:
                return False
        return True

    def heuristic(self, node):
        parent = dict()
        rank = dict()
        unvisited_node_ids = set(map(lambda node: node[0],
                                     self.data['nodes'])) - \
            node.state.data['visited_node_ids']
        total_delay = 0
        for node_id in unvisited_node_ids:
            parent[node_id] = node_id
            rank[node_id] = 0
        possible_edges = []
        for edge in self.data['delay']:
            delay = self.data['delay'][edge]
            if edge <= unvisited_node_ids:
                edge_list = list(edge)
                possible_edges.append((edge_list[0], edge_list[1], delay))
        possible_edges.sort(key=lambda e: e[2])
        for edge in possible_edges:
            node_a, node_b, delay = edge
            node_a_root = node_a
            node_b_root = node_b
            while node_a_root != parent[node_a_root]:
                node_a_root = parent[node_a_root]
            while node_b_root != parent[node_b_root]:
                node_b_root = parent[node_b_root]
            if node_a_root == node_b_root:
                continue
            total_delay += delay
            if rank[node_a_root] > rank[node_b_root]:
                parent[node_b_root] = node_a_root
            else:
                parent[node_a_root] = node_b_root
                if rank[node_a_root] == rank[node_b_root]:
                    rank[node_b_root] += 1
        return total_delay

    def result(self, state, action):
        visited_node_ids = state.data['visited_node_ids'] | {action}
        state_data = {
            'visited_node_ids': visited_node_ids,
            'visited': state.data['visited'] |
            {(frozenset(visited_node_ids), action)}
        }
        return State(name=action, data=state_data)

    def step_cost(self, state, action):
        if not state.name:
            return 0
        return self.data['delay'][frozenset([state.name, action])]


class PancakeProblem(ProblemInterface):
    """The Burnt pancake problem"""

    def _build_initial_state(self, data):
        return State('', {
            'order': self.data['init'],
            'seen': set(self.data['init'])
        })

    def actions(self, state):
        feasible_actions = []
        for flip_idx in range(len(self.data['goal'])):
            flip_result = self.flip_pancakes(state.data['order'], flip_idx)
            if flip_result not in state.data['seen']:
                feasible_actions.append(flip_idx)
        return feasible_actions

    def final_cost(self, node):
        return node.path_cost

    @classmethod
    def flip_pancakes(cls, order, flip_idx):
        """Perform a flip operation on pancakes"""
        new_order = list(order)
        new_order[0:flip_idx + 1] = [-x for x in order[0:flip_idx + 1][::-1]]
        return tuple(new_order)

    def goal_test(self, state):
        return state.data['order'] == self.data['goal']

    def heuristic(self, node):
        breakpoint = 0
        order = node.state.data['order']
        for idx in range(1, len(self.data['goal'])):
            if order[idx] - order[idx - 1] != 1:
                breakpoint += 1
        return breakpoint

    def result(self, state, action):
        flip_result = self.flip_pancakes(state.data['order'], action)
        return State(name='flip top ' + str(action + 1), data={
            'order': flip_result,
            'seen': state.data['seen'] | {flip_result}
        })

    def step_cost(self, state, action):
        return 1


def breadth_first_search(problem):
    """Perform a breadth-first search"""
    node_count = 1
    max_explored_count = 0
    max_frontier_count = 1
    node = Node(problem, problem.initial_state)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    while True:
        if REPORT_MODE:
            elapsed_secs = (datetime.now() - START_TIME).total_seconds()
            if elapsed_secs > REPORT_TIMEOUT_SECS:
                return {
                    'node': Node(problem, State(
                        'Timed out after ' + str(elapsed_secs) + ' sec', None)),
                    'node_count': node_count,
                    'max_explored_count': max_explored_count,
                    'max_frontier_count': max_frontier_count,
                }
        if not frontier:
            return {
                'node': None,
                'node_count': node_count,
                'max_explored_count': max_explored_count,
                'max_frontier_count': max_frontier_count,
            }
        max_frontier_count = max(max_frontier_count, len(frontier))
        node = frontier.popleft()
        explored.add(node.path)
        max_explored_count = max(max_explored_count, len(explored))
        for action in problem.actions(node.state):
            child = node.child_node(problem, action)
            node_count += 1
            if child.path not in (explored |
                                  set(map(lambda node: node.path,
                                          frontier))):
                if problem.goal_test(child.state):
                    return {
                        'node': child,
                        'node_count': node_count,
                        'max_explored_count': max_explored_count,
                        'max_frontier_count': max_frontier_count,
                    }
                frontier.append(child)


def recursive_dls(problem, node, limit):
    """Recursive helper for depth-limited search"""
    node_count = 0
    if REPORT_MODE:
        elapsed_secs = (datetime.now() - START_TIME).total_seconds()
        if elapsed_secs > REPORT_TIMEOUT_SECS:
            return {
                'node': Node(problem, State(
                    'Timed out after ' + str(elapsed_secs) + ' sec', None)),
                'node_count': node_count,
            }
    if problem.goal_test(node.state):
        return {
            'node': node,
            'node_count': node_count,
        }
    if limit == 0:
        return {
            'node': None,
            'node_count': node_count,
        }
    for action in problem.actions(node.state):
        child = node.child_node(problem, action)
        node_count += 1
        result = recursive_dls(problem, child, limit - 1)
        node_count += result['node_count']
        if result['node'] is not None:
            return {
                'node': result['node'],
                'node_count': node_count,
            }
    return {
        'node': None,
        'node_count': node_count,
    }


def iterative_deepening_search(problem):
    """Perform an iterative deepening depth-first search"""
    node_count = 0
    depth = 0

    while True:
        node_count += 1
        init_node = Node(problem, problem.initial_state)
        result = recursive_dls(problem, init_node, depth)
        node_count += result['node_count']
        if result['node'] is not None:
            return {
                'node': result['node'],
                'node_count': node_count,
            }
        depth += 1


def best_first_search(problem, f):
    """Perform a best-first search with given evaludation function"""
    node_count = 1
    max_explored_count = 0
    max_frontier_count = 1
    node = Node(problem, problem.initial_state)
    frontier = [(f(node), node)]
    explored = set()

    if DEBUG_MODE:
        max_explored_length = 0
        prev_time = datetime.now()

    while True:
        if REPORT_MODE:
            elapsed_secs = (datetime.now() - START_TIME).total_seconds()
            if elapsed_secs > REPORT_TIMEOUT_SECS:
                return {
                    'node': Node(problem, State(
                        'Timed out after ' + str(elapsed_secs) + ' sec', None)),
                    'node_count': node_count,
                    'max_explored_count': max_explored_count,
                    'max_frontier_count': max_frontier_count,
                }
        if not frontier:
            return {
                'node': None,
                'node_count': node_count,
                'max_explored_count': max_explored_count,
                'max_frontier_count': max_frontier_count,
            }
        max_frontier_count = max(max_frontier_count, len(frontier))
        node = heappop(frontier)[1]
        if problem.goal_test(node.state):
            return {
                'node': node,
                'node_count': node_count,
                'max_explored_count': max_explored_count,
                'max_frontier_count': max_frontier_count,
            }
        explored.add(node.path)

        if DEBUG_MODE:
            if max(list(map(len, explored))) > max_explored_length:
                curr_time = datetime.now()
                print(curr_time, max_explored_length, '+',
                      (curr_time - prev_time).total_seconds(), 's')
                prev_time = curr_time
                max_explored_length = max(list(map(len, explored)))

        max_explored_count = max(max_explored_count, len(explored))
        for action in problem.actions(node.state):
            child = node.child_node(problem, action)
            child_weight = f(child)
            node_count += 1
            try:
                frontier_idx = list(map(lambda result: result[1].path,
                                        frontier)).index(child.path)
                if f(child) < frontier[frontier_idx][0]:
                    frontier[frontier_idx] = (child_weight, child)
                    heapify(frontier)
            except ValueError:
                if child.path not in explored:
                    heappush(frontier, (child_weight, child))


def uniform_cost_search(problem):
    """Perform a uniform-cost search"""
    return best_first_search(problem,
                             lambda n: n.path_cost)


def greedy_search(problem):
    """Perform a greedy best-first search"""
    return best_first_search(problem, problem.heuristic)


def a_star_search(problem):
    """Perform an A* best-first search"""
    return best_first_search(problem,
                             lambda n: n.path_cost + problem.heuristic(n))


def main():
    """Main function"""
    parser = ArgumentParser(description='Solve search problems.')
    parser.add_argument('config_file', type=FileType('r'),
                        help='the configuration file for the problem')
    parser.add_argument('algorithm', type=str,
                        choices=['bfs', 'unicost', 'greedy', 'iddfs', 'Astar'],
                        help='the search algorithm to use to solve the problem')

    args = parser.parse_args()
    problem_name = args.config_file.readline().strip().lower()
    algorithm_name = args.algorithm

    if REPORT_MODE:
        print('Configuration File:', args.config_file.name)
        print('Problem Name:', problem_name)
        print('Algorithm:', algorithm_name)

    if problem_name == 'monitor':
        sensors = literal_eval(args.config_file.readline())
        targets = literal_eval(args.config_file.readline())
        problem = MonitorProblem({
            'sensors': sensors,
            'targets': targets,
        })

    elif problem_name == 'aggregation':
        nodes = literal_eval(args.config_file.readline())
        delay_dict = {}
        while True:
            delay_line = args.config_file.readline()
            if not delay_line.strip():
                break
            delay = literal_eval(delay_line)
            delay_dict[frozenset([delay[0], delay[1]])] = delay[2]
        problem = AggregationProblem({
            'nodes': nodes,
            'delay': delay_dict,
        })

    elif problem_name == 'pancakes':
        problem = PancakeProblem({
            'init': literal_eval(args.config_file.readline()),
            'goal': literal_eval(args.config_file.readline()),
        })

    else:
        raise SystemExit('error: bad configuration file: problem keyword '
                         '\'' + problem_name + '\' unknown')

    if algorithm_name == 'bfs':
        solution = breadth_first_search(problem)

    elif algorithm_name == 'unicost':
        solution = uniform_cost_search(problem)

    elif algorithm_name == 'iddfs':
        solution = iterative_deepening_search(problem)

    elif algorithm_name == 'greedy':
        solution = greedy_search(problem)

    elif algorithm_name == 'Astar':
        solution = a_star_search(problem)

    else:
        # We should never get here
        raise NotImplementedError('algorithm not implemented')

    curr_node = solution['node']
    if curr_node is None:
        print('No solution')
    else:
        for state_name in curr_node.path:
            if state_name:
                print(state_name)
    print('Time:', solution['node_count'])
    if 'max_frontier_count' in solution and 'max_explored_count' in solution:
        print('Space: Frontier', str(solution['max_frontier_count']) + ',',
              'Visited', solution['max_explored_count'])
    else:
        print('Space: Not available for tree search')
    if curr_node is None:
        raise SystemExit()
    print('Cost:', problem.final_cost(solution['node']))
    if REPORT_MODE:
        print('-' * 80)


if __name__ == '__main__':
    main()
