
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.
    """
    start_alpha = arm.getArmAngle()[0]
    alpha_min = arm.getArmLimit()[0][0]
    alpha_max = arm.getArmLimit()[0][1]
    
    start_beta = arm.getArmAngle()[1]
    beta_min = arm.getArmLimit()[1][0]
    beta_max = arm.getArmLimit()[1][1]
    
    rows = int(((alpha_max - alpha_min)/granularity)+1)
    columns = int(((beta_max - beta_min)/granularity)+1)
    maze = [[SPACE_CHAR for _ in range(columns)] for _ in range(rows)]
    
    alpha = alpha_min
    offsets = (alpha_min, beta_min)

    while alpha <= alpha_max:
        beta = beta_min
        while beta <= beta_max:
            
            alpha_index = angleToIdx([alpha], [offsets[0]], granularity)[0]
            beta_index = angleToIdx([beta], [offsets[1]], granularity)[0]
            arm.setArmAngle((alpha, beta))
            
            if alpha == start_alpha and beta == start_beta:
                maze[alpha_index][beta_index] = START_CHAR
            elif not isArmWithinWindow(arm.getArmPos(), window):
                maze[alpha_index][beta_index] = WALL_CHAR
            elif doesArmTouchObjects(arm.getArmPosDist(), obstacles, False):
                maze[alpha_index][beta_index] = WALL_CHAR
            elif doesArmTouchObjects(arm.getArmPosDist(), goals, True):
                if doesArmTipTouchGoals(arm.getEnd(), goals):
                    maze[alpha_index][beta_index] = OBJECTIVE_CHAR
                else:
                    maze[alpha_index][beta_index] = WALL_CHAR
            
            beta += granularity
        alpha += granularity
    
    return Maze(maze, offsets, granularity)

            



















