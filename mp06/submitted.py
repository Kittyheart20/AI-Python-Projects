# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

import queue

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    q = queue.Queue()
    s = set([maze.start])
    backpath = {}
    path = []

    q.put(maze.start)

    while (q.empty() == False):
        cell = q.get()
        #print(cell)
        if (cell == maze.waypoints[0]):
            #Reconstuct and return
            currcell = cell
            while (currcell != maze.start):
                path.append(currcell)
                currcell = backpath[currcell]
            path.append(maze.start)
            path.reverse()
            break

        for neighbor in maze.neighbors_all(cell[0], cell[1]):
            if (neighbor not in s):
                q.put(neighbor)
                s.add(neighbor)
                backpath[neighbor] = cell

    #TODO: Implement bfs function

    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.
or ((gvals.get(neighbor) != None) and (g < gvals.get(neighbor)))
    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    q = queue.PriorityQueue()
    visited = set([maze.start])
    gvals = {}
    backpath = {}
    path = []

    q.put((0, maze.start))
    gvals[maze.start] = 0

    while (q.empty() == False):
        cell = q.get()[1]
        #print(cell)
        if (cell == maze.waypoints[0]):
            #Reconstuct and return
            currcell = cell
            while (currcell != maze.start):
                path.append(currcell)
                currcell = backpath[currcell]
            path.append(maze.start)
            path.reverse()
            break
        
        for neighbor in maze.neighbors_all(cell[0], cell[1]): 
            g = gvals[cell] + 1
            h = max(abs(maze.waypoints[0][0] - neighbor[0]), abs(maze.waypoints[0][1] - neighbor[1])) 
            if (neighbor not in visited or ((gvals.get(neighbor) != None) and (g < gvals.get(neighbor)))):
                visited.add(neighbor)
                q.put((g+h, neighbor))
                backpath[neighbor] = cell
                gvals[neighbor] = g
                      

    return path

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    return []
