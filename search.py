# search.py
# ---------


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import csv
import os
import time

import util


class _TraceLogger:
    columns = ('iteration', 'expanded_state', 'parent', 'action',
               'generated_successors', 'frontier_before', 'frontier_after',
               'explored', 'g', 'h', 'f')

    def __init__(self, algorithm):
        os.makedirs('evidence', exist_ok=True)
        filename = '{}_{}.csv'.format(algorithm, time.time_ns())
        self.handle = open(os.path.join('evidence', filename), 'w', newline='')
        self.writer = csv.DictWriter(self.handle, fieldnames=self.columns)
        self.writer.writeheader()

    def record(self, iteration, state, parent, action, successors,
               frontier_before, frontier_after, explored, cost, heuristic):
        self.writer.writerow({
            'iteration': iteration,
            'expanded_state': repr(state),
            'parent': repr(parent),
            'action': repr(action),
            'generated_successors': repr(successors),
            'frontier_before': frontier_before,
            'frontier_after': frontier_after,
            'explored': len(explored),
            'g': cost,
            'h': heuristic,
            'f': cost + heuristic,
        })
        self.handle.flush()

    def close(self):
        self.handle.close()


def _trace_return(logger, result):
    logger.close()
    return result

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    start = problem.getStartState()
    stack = util.Stack()
    stack.push((start, [], None, None))
    visited = set()
    logger = _TraceLogger('dfs')
    iteration = 0

    while not stack.isEmpty():
        frontier_before = len(stack.list)
        state, path, parent, action = stack.pop()
        if state in visited:
            continue
        visited.add(state)

        if problem.isGoalState(state):
            return _trace_return(logger, path)

        successors = problem.getSuccessors(state)
        for successor, next_action, _ in successors:
            if successor not in visited:
                stack.push((successor, path + [next_action], state, next_action))
        logger.record(iteration, state, parent, action, successors,
                      frontier_before, len(stack.list), visited, len(path), 0)
        iteration += 1

    return _trace_return(logger, None)

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    start = problem.getStartState()
    queue = util.Queue()
    queue.push((start, [], None, None))
    visited = set([start])
    logger = _TraceLogger('bfs')
    iteration = 0

    while not queue.isEmpty():
        frontier_before = len(queue.list)
        state, path, parent, action = queue.pop()
        if problem.isGoalState(state):
            return _trace_return(logger, path)

        successors = problem.getSuccessors(state)
        for successor, next_action, _ in successors:
            if successor not in visited:
                visited.add(successor)
                queue.push((successor, path + [next_action], state, next_action))
        logger.record(iteration, state, parent, action, successors,
                      frontier_before, len(queue.list), visited, len(path), 0)
        iteration += 1

    return _trace_return(logger, None)

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    start = problem.getStartState()
    priorityQueue = util.PriorityQueue()
    priorityQueue.push((start, [], None, None), 0)
    bestCost = {start: 0}
    logger = _TraceLogger('ucs')
    iteration = 0

    while not priorityQueue.isEmpty():
        frontier_before = len(priorityQueue.heap)
        state, path, parent, action = priorityQueue.pop()
        currentCost = bestCost.get(state, float('inf'))

        if problem.isGoalState(state):
            return _trace_return(logger, path)

        successors = problem.getSuccessors(state)
        for successor, next_action, stepCost in successors:
            nextCost = currentCost + stepCost
            if nextCost < bestCost.get(successor, float('inf')):
                bestCost[successor] = nextCost
                priorityQueue.push((successor, path + [next_action], state, next_action), nextCost)
        logger.record(iteration, state, parent, action, successors,
                      frontier_before, len(priorityQueue.heap), bestCost,
                      currentCost, 0)
        iteration += 1

    return _trace_return(logger, None)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def greedyBestFirstSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search using only h(n), without including the accumulated path cost."""
    start = problem.getStartState()
    priorityQueue = util.PriorityQueue()
    priorityQueue.push((start, [], None, None), heuristic(start, problem))
    visited = set()
    logger = _TraceLogger('gbfs')
    iteration = 0

    while not priorityQueue.isEmpty():
        frontier_before = len(priorityQueue.heap)
        state, path, parent, action = priorityQueue.pop()
        if state in visited:
            continue
        visited.add(state)
        if problem.isGoalState(state):
            return _trace_return(logger, path)

        successors = problem.getSuccessors(state)
        for successor, next_action, _ in successors:
            if successor not in visited:
                next_path = path + [next_action]
                priorityQueue.push((successor, next_path, state, next_action),
                                   heuristic(successor, problem))
        h_value = heuristic(state, problem)
        logger.record(iteration, state, parent, action, successors,
                      frontier_before, len(priorityQueue.heap), visited,
                      len(path), h_value)
        iteration += 1

    return _trace_return(logger, None)

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start = problem.getStartState()
    priorityQueue = util.PriorityQueue()
    priorityQueue.push((start, [], None, None), heuristic(start, problem))
    bestCost = {start: 0}
    logger = _TraceLogger('astar')
    iteration = 0

    while not priorityQueue.isEmpty():
        frontier_before = len(priorityQueue.heap)
        state, path, parent, action = priorityQueue.pop()
        currentCost = bestCost.get(state, float('inf'))

        if problem.isGoalState(state):
            return _trace_return(logger, path)

        successors = problem.getSuccessors(state)
        for successor, next_action, stepCost in successors:
            nextCost = currentCost + stepCost
            if nextCost < bestCost.get(successor, float('inf')):
                bestCost[successor] = nextCost
                priorityQueue.push((successor, path + [next_action], state, next_action),
                                   nextCost + heuristic(successor, problem))
        h_value = heuristic(state, problem)
        logger.record(iteration, state, parent, action, successors,
                      frontier_before, len(priorityQueue.heap), bestCost,
                      currentCost, h_value)
        iteration += 1

    return _trace_return(logger, None)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
gbfs = greedyBestFirstSearch
