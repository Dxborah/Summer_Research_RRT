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
        self.nodeRad = 2
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
        '''
        obstacles_list = obstacles.copy()
        while (len(obstacles_list) > 0):
            obstacle = obstacles_list.pop(0)
            pygame.draw.rect(self.map, self.grey, obstacle)
        '''
        for obstacle in obstacles:
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
        self.obstacles = obs     #obs.copy()
        return obs


    def add_node(self, n, x, y):
        self.x.insert(n, x) #n is the indice in the list of nodes
        self.y.append(y)

    def remove_node(self, n):
        self.x.pop(n)
        self.y.pop(n)

    def add_edge(self, parent, child):
        self.parent.insert(child, parent) # child:parent 

    def remove_edge(self, n):
        self.parent.pop(n) #n is index of child

    def number_of_nodes(self):
        return len(self.x) 

    #euclidean distance
    def distance(self, n1, n2):
        (x1, y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])
        return math.hypot(x2 - x1, y2 - y1)

    #generates random samples from map
    def sample_envir(self):
        x = int(random.uniform(0, self.Mapw))
        y = int(random.uniform(0, self.Maph))
        return x,y

    #
    def nearest(self, n):
        dmin = self.distance(0, n)
        nnear = 0
        for i in range(0, n):
            if self.distance(i, n) < dmin:
                dmin = self.distance(i, n)
                nnear = i
        return nnear

    #makes sure node is in the free space
    def isFree(self):
        '''
        n = self.number_of_nodes() -1 # index of most recent node
        (x, y) = (self.x[n], self.y[n])
        obs = self.obstacles.copy()
        while len(obs) > 0:
            rectangle = obs.pop(0)
            if rectangle.collidepoint(x, y): #Checks if the most recent node’s point (x, y) is inside the current obstacle rectangle
                self.remove_node(n)
                return False
        return True
        '''
        n = self.number_of_nodes() - 1  # index of most recent node
        (x, y) = (self.x[n], self.y[n])

        for rectangle in self.obstacles:
            if rectangle.collidepoint(x, y):  # Check if the node is inside an obstacle
                self.remove_node(n)
                return False
        return True

    #finds if and edge (connection between 2 nodes) crosses any obstacles
    def crossObstacle(self, x1, x2, y1, y2):
        '''
        obs = self.obstacles.copy()
        while(len(obs) > 0):
            rectangle = obs.pop(0)
            for i in range(0, 101):
                u = i/100 #parameter ranging from 0.0 to 1.0 in steps of 0.01
                x = x1*u + x2*(1-u) #computes an interpolated point along the line segment
                y = y1*u + y2*(1-u)

                #If any interpolated point lies within an obstacle, the edge intersects an obstacle 
                if rectangle.collidepoint(x, y):
                    return True 
        return False
        '''
        if x1 == x2 and y1 == y2:
            return False
        
        for rectangle in self.obstacles:
            
            for i in range(0, 21):
                u = i / 20  # step size
                x = x1 * u + x2 * (1 - u)
                y = y1 * u + y2 * (1 - u)

                if rectangle.collidepoint(x, y):
                    return True  # Edge crosses an obstacle
        return False  # No collision
    
    # connects n1 and n2 if no obstacle between them
    def connect(self, n1, n2):
        (x1, y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])
        if self.crossObstacle(x1, x2, y1, y2):
            self.remove_node(n2)
            return False
        else:
            self.add_edge(n1, n2)
            return True

    def step(self, near, nrand, dmax=35):
        d = self

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