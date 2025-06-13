import pygame
from pygame_version.RRTbase import RRTGraph
from pygame_version.RRTbase import RRTMap

def main():
    dimensions = (600, 1000)
    start = (50, 50)
    goal = (510, 510)
    obsdim = 30
    obsnum = 4
    iter = 0

    pygame.init()
    rrt_map = RRTMap(start, goal, dimensions, obsdim, obsnum)
    rrt_graph = RRTGraph(start, goal, dimensions, obsdim, obsnum)
    
    obstacles = rrt_graph.makeObs()
    rrt_map.drawMap(obstacles)

    while(True):
        x, y = rrt_graph.sample_envir()
        n = rrt_graph.number_of_nodes()
        rrt_graph.add_node(n, x, y)
        rrt_graph.add_edge(n-1, n)
        x1, y1 = rrt_graph.x[n], rrt_graph.y[n]
        x2, y2 = rrt_graph.x[n-1], rrt_graph.y[n-1]
        if(rrt_graph.isFree()):
            pygame.draw.circle(rrt_map.map, rrt_map.red, (rrt_graph.x[n], rrt_graph.y[n]), rrt_map.nodeRad,rrt_map.nodeThickness)
            if not rrt_graph.crossObstacle(x1, x2, y1, y2):
                pygame.draw.line(rrt_map.map, rrt_map.blue, (x1, y1), (x2, y2), rrt_map.edgeThickness)

    pygame.display.update()
    pygame.event.clear()
    pygame.event.wait(0)

if __name__ == '__main__':
    main()