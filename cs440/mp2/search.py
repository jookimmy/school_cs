# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def backtrace(node, start, parent):
    path = []
    while node != start:
        path.append(node)
        node = parent[node]
    path.append(start)
    return path[::-1]

def bfs(maze):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 
    """
    queue, visited, parent = list(), set(), dict()
    start = maze.getStart()
    goals = maze.getObjectives()

    queue.append(start)
    visited.add(start)
    parent[start] = None

    while queue:
        curr = queue.pop(0)
        if curr in goals:
            return backtrace(curr, start, parent)
        for node in maze.getNeighbors(curr[0],curr[1]):
            if node not in visited:
                queue.append(node)
                visited.add(node)
                parent[node] = curr

    return None








