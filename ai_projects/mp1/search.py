# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)
import heapq
from queue import PriorityQueue
import copy
def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "extra": extra,
    }.get(searchMethod)(maze)

def backtrace(node,start,parent):
    path = []
    while node != start:
        path.append(node)
        node = parent[node]
    path.append(start)
    return path[::-1]

def bfs_helper(start, goal, maze):
    queue = []
    # dont think I need a visited if I can just check from parents
    visited, parent = set(), dict()
    queue.append(start)
    visited.add(start)
    while queue:
        curr = queue.pop(0)
        if curr == goal:
            # calls on backtrace function for returning path
            return backtrace(curr, start, parent)
        for node in maze.getNeighbors(curr[0],curr[1]):
            if node not in visited:
                queue.append(node)
                visited.add(node)
                parent[node] = curr
    return []

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    goal = maze.getObjectives()[0]
    return bfs_helper(start, goal, maze)

def astar_helper(start, goal, maze):
    # TODO: Write your code here
    # trying a heapq for this one
    # heap = []
    queue = PriorityQueue()
    parent, cost_map, visited = dict(), dict(), set()
    # intializing heap with initial node
    # heapq.heappush(heap, (0, start))
    queue.put((0, start))
    visited.add(start)
    cost_map[start] = 0
    parent[start] = None
    while queue:
        # curr = heapq.heappop(heap)
        curr = queue.get()
        # setting the cost or g value 
        cost = curr[0] + 1
        # coordinates for the current node in heap
        coord = (curr[1][0], curr[1][1])
        if coord == goal:
            # calls on backtrace function for returning path
            return backtrace(coord, start, parent)
        for node in maze.getNeighbors(coord[0],coord[1]):
            # checking if node is not visited or if there is a smaller cost of getting to the node
            if node not in visited or cost < cost_map[node]:
                if node not in visited:
                    heuristic = manhattan_dist(goal, node)
                    # heapq.heappush(heap, (cost + heuristic, node))
                    queue.put((cost + heuristic, node))
                    visited.add(node)
                parent[node] = coord
                cost_map[node] = cost
    return []

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    goal = (maze.getObjectives()[0][0], maze.getObjectives()[0][1])
    return astar_helper(start, goal, maze)

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.
        
    @param maze: The maze to execute the search on.
        
    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    # going to try priorityQueue this time
    frontier = PriorityQueue()
    parent, cost_map, visited = dict(), dict(), set()
    
    # initializing start point, objectives, and table of astar distances
    start = maze.getStart()
    goals = maze.getObjectives()
    dist_table = compute_distances(maze, goals)
    # find closest goal then use to get minimum spanning tree sum for initial no objectives reached
    # find h by adding manhattendistance and mst heuristic
    closest_goal = find_closest(start, goals)
    mst_sum = mst(goals, dist_table)
    mst_dict = {tuple(sorted(goals)) : mst_sum}
    h = manhattan_dist(closest_goal, start) + mst_sum

    # initializing frontier and all status objects
    start_state = (start, tuple(sorted(goals)))
    frontier.put((h, start_state))
    visited.add(start_state)
    cost_map[start_state] = 0
    parent[start_state] = None
    # path = []
    # order = []
    while frontier:
        curr_state = frontier.get()[1]
        coord = curr_state[0]
        curr_goals = curr_state[1]
        cost = cost_map[curr_state] + 1

        if coord in curr_goals:
            # print(goals)
            objs = set(curr_goals)
            objs.remove(coord)
            curr_goals = tuple(sorted(objs))
            # print(goals)
            # order.append(curr_state)
            # frontier.queue.clear()
            if len(objs) == 0:
                path = backtrace_multi(curr_state, start_state, parent)
                path.insert(0, start)
                return path

        closest_goal = find_closest(coord, curr_goals)
        s_goals = tuple(sorted(curr_goals))
        if s_goals not in mst_dict:
            mst_sum = mst(curr_goals, dist_table)
            mst_dict[s_goals] = mst_sum

        for node in maze.getNeighbors(coord[0], coord[1]):
            next_state = (node, s_goals)
            if next_state not in visited or cost < cost_map[next_state]:
                # if next_state not in visited:
                heuristic = manhattan_dist(closest_goal, node) + mst_sum
                frontier.put((cost + heuristic, next_state))
                visited.add(next_state)
                # if parent[curr_state] == start_state:
                #     print(next_state)
                #     print(curr_state)
                # print(curr_state)
                parent[next_state] = curr_state
                cost_map[next_state] = cost
    # order.insert(0, start_state)
    # for i in range(len(order) - 1):
        # print(backtrace_multi(order[i+1], order[i], parent))
    # path.insert(0, start)
    # print(parent)
    return []


def backtrace_multi(node_state, start_state, parent):
    path = []
    while node_state != start_state:
        # print(node_state, start_state)
        path.append(node_state[0])
        node_state = parent[node_state]
    # print(node_state, start_state)
    return path[::-1]

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    frontier = PriorityQueue()
    parent, cost_map, visited = dict(), dict(), set()
    
    # initializing start point, objectives, and table of astar distances
    start = maze.getStart()
    goals = maze.getObjectives()
    new_maze = copy.deepcopy(maze)
    dist_table = compute_distances(new_maze, goals)
    # find closest goal then use to get minimum spanning tree sum for initial no objectives reached
    # find h by adding manhattendistance and mst heuristic
    closest_goal = find_closest(start, goals)
    mst_sum = mst(goals, dist_table)
    mst_dict = {tuple(sorted(goals)) : mst_sum}
    h = manhattan_dist(closest_goal, start) + mst_sum

    # initializing frontier and all status objects
    start_state = (start, tuple(sorted(goals)))
    frontier.put((h, start_state))
    visited.add(start_state)
    cost_map[start_state] = 0
    parent[start_state] = None
    curr_goals = goals
    # path = []
    goals_left = len(goals)
    # order = []
    while len(curr_goals) > 0 :
        curr_state = frontier.get()[-1]
        coord = curr_state[0]
        curr_goals = curr_state[1]
        cost = cost_map[curr_state] + 1

        objs = set(curr_goals)
        if coord in objs:
            goals_left -= 1
            # print(goals)
            objs.remove(coord)
            curr_goals = tuple(sorted(objs))
            # print(goals)
            # order.append(curr_state)
            # for state in frontier.queue:
            #     if len(state[-1]) != goals_left:
            #         frontier.queue.remove(state)

            if len(objs) == 0:
                path = backtrace_multi(curr_state, start_state, parent)
                path.insert(0, start)
                return path

        closest_goal = find_closest(coord, curr_goals)
        s_goals = tuple(sorted(curr_goals))
        
        if s_goals not in mst_dict:
            mst_sum = mst(curr_goals, dist_table)
            mst_dict[s_goals] = mst_sum

        for node in maze.getNeighbors(coord[0], coord[1]):
            next_state = (node, s_goals)
            if next_state not in visited or cost < cost_map[next_state]:
                # if next_state not in visited:
                if next_state not in visited:
                    heuristic = manhattan_dist(closest_goal, node) + mst_sum
                    frontier.put((cost + heuristic, cost, next_state))
                    visited.add(next_state)
                # if parent[curr_state] == start_state:
                #     print(next_state)
                #     print(curr_state)
                # print(curr_state)
                parent[next_state] = curr_state
                cost_map[next_state] = cost
    # order.insert(0, start_state)
    # for i in range(len(order) - 1):
        # print(backtrace_multi(order[i+1], order[i], parent))
    # path.insert(0, start)
    # print(parent)
    return []

def mst(objectives, dist_table):
    # prims algorithm to calculate mst for remaining objectives
    if len(objectives) == 1:
        return 0
    # creating sets/edge costs for minimum spanning tree
    remaining, mst, weights = set(objectives), list(), {obj: float('inf') for obj in objectives}
    # initializing weight for first vertex
    # get a random current state to be replaced by lowest cost later
    start = list(remaining)[0]
    weights[start] = 0
    while remaining:
        # find minimum key value
        curr_state = list(remaining)[0]
        for obj in remaining:
            if weights[obj] < weights[curr_state]:
                curr_state = obj
        # add to minimum spanning tree
        mst.append(curr_state)
        # check each vertex cost to replace them with a* table cost
        for vertex in objectives:
            if curr_state == vertex:
                continue
            #find cost
            cost = dist_table[curr_state][vertex]
            if cost < weights[vertex]:
                # print(f'Cost: {cost} | Weight: {weights[vertex]} | Current State: {vertex}')
                # print(cost, weights[vertex], (vertex, curr_state))
                weights[vertex] = cost
                # print(cost, weights[vertex], (vertex, curr_state))
        # remove node from mst cause visited
        remaining.remove(curr_state)
    return sum(weights[node] for node in mst)

def manhattan_dist(pt1, pt2):
    return abs(pt2[1]-pt1[1]) + abs(pt2[0]-pt1[0])

def find_closest(pt1, objs):
    closest = objs[0]
    c_distance = float('inf')
    for obj in objs:
        new_dist = manhattan_dist(pt1, closest)
        if new_dist < c_distance:
            closest = obj
            c_distance = new_dist
    return closest

def compute_distances(maze, goals):
    dist = dict()
    for i in goals:
        dist[i] = None
    for start in goals:
        dist[start] = dict()
        for goal in goals:
            if start != goal:
                # path = astar_helper(start, goal, maze)
                # dist[start][goal] = (len(path) - 1)
                new_maze = copy.deepcopy(maze)
                dist[start][goal] = len(astar_helper(start,goal,new_maze)) - 1
                # dist[start][goal] = len(bfs_helper(start,goal,maze)) - 1
                # dist[start][goal] = manhattan_dist(start,goal)
    return dist

def extra(maze):
    """
    Runs extra credit suggestion.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    frontier = PriorityQueue()
    parent, cost_map, visited = dict(), dict(), set()
    
    # initializing start point, objectives, and table of astar distances
    start = maze.getStart()
    goals = maze.getObjectives()
    new_maze = copy.deepcopy(maze)
    dist_table = compute_distances(new_maze, goals)
    # find closest goal then use to get minimum spanning tree sum for initial no objectives reached
    # find h by adding manhattendistance and mst heuristic
    closest_goal = find_closest(start, goals)
    mst_sum = mst(goals, dist_table)
    mst_dict = {tuple(sorted(goals)) : mst_sum}
    h = manhattan_dist(closest_goal, start) + mst_sum

    # initializing frontier and all status objects
    start_state = (start, tuple(sorted(goals)))
    frontier.put((h, 0, start_state))
    visited.add(start_state)
    cost_map[start_state] = 0
    parent[start_state] = None
    curr_goals = goals
    # path = []
    goals_left = len(goals)
    # order = []
    while len(curr_goals) > 0 :
        curr_state = frontier.get()[-1]
        coord = curr_state[0]
        curr_goals = curr_state[1]
        cost = cost_map[curr_state] + 1

        objs = set(curr_goals)
        if coord in objs:
            goals_left -= 1
            # print(goals)
            objs.remove(coord)
            curr_goals = tuple(sorted(objs))
            # print(goals)
            # order.append(curr_state)
            for state in frontier.queue:
                if len(state[-1]) != goals_left:
                    frontier.queue.remove(state)

            if len(objs) == 0:
                path = backtrace_multi(curr_state, start_state, parent)
                path.insert(0, start)
                return path

        closest_goal = find_closest(coord, curr_goals)
        s_goals = tuple(sorted(curr_goals))
        
        if s_goals not in mst_dict:
            mst_sum = mst(curr_goals, dist_table)
            mst_dict[s_goals] = mst_sum

        for node in maze.getNeighbors(coord[0], coord[1]):
            next_state = (node, s_goals)
            if next_state not in visited or cost < cost_map[next_state]:
                # if next_state not in visited:
                if next_state not in visited:
                    heuristic = manhattan_dist(closest_goal, node) + (1.5 * mst_sum)
                    frontier.put((cost + heuristic, cost, next_state))
                    visited.add(next_state)
                # if parent[curr_state] == start_state:
                #     print(next_state)
                #     print(curr_state)
                # print(curr_state)
                parent[next_state] = curr_state
                cost_map[next_state] = cost
    # order.insert(0, start_state)
    # for i in range(len(order) - 1):
        # print(backtrace_multi(order[i+1], order[i], parent))
    # path.insert(0, start)
    # print(parent)
    return []
