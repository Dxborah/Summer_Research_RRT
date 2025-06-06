import pygame
from RRTbase import RRTGraph
from RRTbase import RRTMap

def main():
    dimensions = (600, 1000)
    start = (50, 50)
    goal = (510, 510)
    obsdim = 30
    obsnum = 50

    pygame.init()
    rrt_map = RRTMap(start, goal, dimensions, obsdim, obsnum)
    rrt_graph = RRTGraph(start, goal, dimensions, obsdim, obsnum)
    
    obstacles = rrt_graph.makeObs()
    rrt_map.drawMap(obstacles)

    pygame.display.update()
    pygame.event.clear()
    pygame.event.wait(0)

if __name__ == '__main__':
    main()