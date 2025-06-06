import random
import pygame
import math

class RRTMap:
    def __init__(self, start, goal, MapDim, obsdim, obsnum):
        self.start = start
        self.goal = goal
        self.MapDim = MapDim
        self.Maph, self.Mapw = self.MapDim

        #window settings
        self.MapWindwName = 'RRT Path Planning'
        pygame.display.set_caption(self.MapWindwName)
        self.map = pygame.display.set_mode((self.Mapw, self.Maph))
        self.map.fill((255, 255, 255))
        self.nodeRad = 0
        self.nodeThickness = 0
        self.edgeThickness =1

        self.obstacles = []
        self.obsdim = obsdim
        self.obsnum = obsnum

        #colors
        self.grey = (70, 70, 70)
        self.blue = (0, 0, 255)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.white = (255, 255, 255)
        
    def drawMap(self, obstacles):
        pygame.draw.circle(self.map, self.green, self.start, self.nodeRad+5, 0)
        pygame.draw.circle(self.map, self.red, self.goal, self.nodeRad+20, 0)
        self.drawObs(obstacles)

    def drawPath(self):
        pass

    def drawObs(self, obstacles):
        obstacles_list = obstacles.copy()
        while (len(obstacles_list) > 0):
            obstacle = obstacles_list.pop(0)
            pygame.draw.rect(self.map, self.grey, obstacle)

class RRTGraph:
    def __init__(self, start, goal, MapDim, obsdim, obsnum):
        (x, y) = start
        self.start = start
        self.goal = goal
        self.goalFlag = False

        self.MapDim = MapDim
        self.Maph, self.Mapw = self.MapDim
        self.x = []
        self.y = []
        self.parent = []

        #initialize tree
        self.x.append(x)
        self.y.append(y)
        self.parent.append(0)

        #obstacles
        self.obstacles = []
        self.obsdim = obsdim
        self.obsnum = obsnum

        #path
        self.goalstate = None
        self.path = []

    #generate corners of obstacles within bounds of Map
    def makeRandomRect(self):
        uppercornerx = int(random.uniform(0, self.Mapw-self.obsdim))
        uppercornery = int(random.uniform(0, self.Maph-self.obsdim))

        return (uppercornerx, uppercornery)

    #create obstacles
    def makeObs(self):
        obs = []

        for i in range(0, self.obsnum):
            rectangle = None
            start_goal_color = True
            while start_goal_color:
                upper = self.makeRandomRect()
                rectangle = pygame.Rect(upper, (self.obsdim, self.obsdim))
                if rectangle.collidepoint(self.start) or rectangle.collidepoint(self.goal):
                    start_goal_color = True
                else:
                    start_goal_color = False
            obs.append(rectangle)
        self.obstacles = obs.copy()
        return obs

    def add_node(self):
        pass

    def remove_node(self):
        pass

    def add_edge(self):
        pass

    def remove_edge(self):
        pass

    def number_of_nodes(self):
        pass

    def distance(self):
        pass

    def nearest(self):
        pass

    def isFree(self):
        pass

    def crossObstacle(self):
        pass

    def connect(self):
        pass

    def step(self):
        pass

    def path_to_goal(self):
        pass

    def getPathCoords(self):
        pass

    def bias(self):
        pass

    def expand(self):
        pass

    def cost(self):
        pass