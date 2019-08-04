#!/usr/bin/env python3
# -*- coding: iso-8859-15 -*-

from os.path import isfile
from sys import exit, argv
from json import load
import numpy as np
from numpy import pi, sin, cos, sqrt
import pygame
from draw import create_image, html2rgb


# For displaying data on screen
status_dict = {True: 'ON',
               False: 'OFF'}


#############################
# Load parameters from file #
#############################

def load_params(file):
    with open(file, 'r') as jsonfile:
        params = load(jsonfile)

    # Define params as global
    global w, h
    global min_x, max_x, min_y, max_y
    global min_x0, max_x0, min_y0, max_y0
    global molecules, walls, grid

    # Screen parameters
    w = params['screen']['width']
    h = params['screen']['height']

    # Area parameters
    min_x = params['area']['min_x']
    max_x = params['area']['max_x']
    min_y = params['area']['min_y']
    max_y = params['area']['max_y']

    # Initial distribution
    # area of particles
    min_x0 = params['init']['min_x']
    max_x0 = params['init']['max_x']
    min_y0 = params['init']['min_y']
    max_y0 = params['init']['max_y']

    # Molecules
    molecules = []
    for group in params['molecules']:
        molecules += [Particle(vel = np.random.uniform(-1, 1, 2),
                               mass = group['mass'],
                               radius = group['radius'],
                               color = group['color'])
                      for i in range(group['num_particles'])]

    # Distribute molecules in init
    # area with no overlaps
    distribute_no_overlaps(molecules,
                           min_x, min_y+550,
                           max_x, max_y)

    # Set initial kinetic energy
    kinetic_energy = params['kinetic_energy']
    Ek = np.sum([m.get_kinetic_energy() for m in molecules])
    for m in molecules:
        m.set_kinetic_energy(kinetic_energy / num_particles)

    # Walls
    walls = []
    for wall in params['walls']:
        start = np.array(wall['start'])
        end = np.array(wall['end'])
        width = np.array(wall['width'])
        color = html2rgb(wall['color'])
        walls.append(Wall(start, end, width, color))

    # Grid
    grid = Grid(params['grid']['Nx'], params['grid']['Ny'],
                min_x, min_y,
                max_x, max_y)



###########################
# General maths functions #
###########################

def normalize(vec):
    """
    Returns normalized vector
    """
    L = np.linalg.norm(vec)
    if L != 0:
        return vec/L
    else:
        return vec*0.0

def scale_vec(vec, size):
    """
    Returns scaled to size
    """
    new_vec = normalize(vec)
    return new_vec * size

def get_angle(vec):
    angle = np.arctan2(vec[1], vec[0])
    return np.degrees(angle)

def rotate(vec, angle):
    """
    returns vector rotated by angle
    """
    c = cos(angle)
    s = sin(angle)
    mat = np.array([[c, -s],
                    [s,  c]])
    return np.dot(mat, vec)

def intersection(x1, x2, x3, x4,
                 y1, y2, y3, y4):
    """
    returns the intersection
    point of two line segments.
    (used for particle-wall interaction)
    """
    a = ((y3-y4)*(x1-x3) + (x4-x3)*(y1-y3))
    b = ((y1-y2)*(x1-x3) + (x2-x1)*(y1-y3))
    c = ((x4-x3)*(y1-y2) - (x1-x2)*(y4-y3))
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    if c != 0.0:
        """
        c = 0 means that the
        intersection point exists.
        """
        return a/c, b/c, p1 + (p2-p1)*(a/c)
    else:
        return 0, 0, np.zeros(2)

def dist(p1, p2):
    """
    returns the Euclidean distance
    between two points p1, p2
    """
    return np.linalg.norm(p2-p1)


###########
# Classes #
###########

class TextOnScreen:
    """
    A class to display a text
    on screen, at a chosen location,
    font, size, etc.
    """
    def __init__(self, pos=(0,0), color=(0, 200, 0),
                       font='Cabin', size=15, text='',
                       centered=False):
        self.pos = pos
        self.color = color
        self.font = pygame.font.SysFont(font, size)
        self.text = text
        self.centered = centered

    def set_text(self, text):
        self.text = text

    def display(self, surface):
        render = self.font.render(self.text,
                                  False,
                                  self.color)
        if self.centered:
            text_rect = render.get_rect(center=(w/2, h/2))
            surface.blit(render, text_rect)
        else:
            surface.blit(render, self.pos)


class Particle:
    """
    A particle with position, velocity, mass
    and a radius. Moves according to Newton's
    laws of motion, including GRAVITY.
    It also exist in a cell, to reduce
    computational costs (i.e. "neighbor lists")
    """
    def __init__(self, id=-1,
                 pos=np.zeros(2),
                 vel=np.zeros(2),
                 mass=1,
                 radius=5,
                 Gravity=False,
                 color='blue'):

        self.id = id

        self.pos = pos
        self.vel = vel
        self.speed = np.linalg.norm(self.vel)
        self.mass = mass
        self.radius = radius
        self.Gravity = Gravity

        self.color = color
        self.selected = False
        if not isfile('images/{}_{}.png'.format(int(radius), color)):
            create_image(color, radius)
            create_image('FFFFFF', radius)
        self.image = pygame.image.load('images/{}_{}.png'.format(int(radius), color))

        self.cell = (-1, -1)
        self.neighbors = []

    def set_cell(self, x, y):
        self.cell = (x, y)

    def set_neighors(self, grid):
        # Reset current neighbors list
        self.neighbors = []

        # Create new list
        x, y = self.cell
        Nx, Ny = grid.Nx, grid.Ny
        neighbors = [grid.objects[i][j] for i in range(x-1, x+2) if 0 <= i < Nx
                                        for j in range(y-1, y+2) if 0 <= j < Ny]
        self.neighbors = [object for sublist in neighbors
                                 for object in sublist
                                 if object is not self]


    def select(self):
        self.selected = True
        self.image = pygame.image.load('images/{}_FFFFFF.png'.format(self.radius))

    def unselect(self):
        self.selected = False
        self.image = pygame.image.load('images/{}_{}.png'.format(self.radius, self.color))

    def flip_selection_status(self):
        if self.selected:
            self.unselect()
        else:
            self.select()

    def gravity(self, yref, dt):
        r2 = (yref - self.pos[1])**2
        g_acc = G / r2 * np.array([0, 1])
        self.add_acceleration(g_acc, dt)

    def add_acceleration(self, a, dt):
        self.vel += a*dt

    def move(self, dt):
        self.pos += self.vel * dt
        self.speed = np.linalg.norm(self.vel)

    def get_potential_grav_energy(self, yref):
        r2 = (yref - self.pos[1])**2
        return G * self.mass / r2

    def get_kinetic_energy(self):
        return 0.5*self.mass*np.linalg.norm(self.vel)**2

    def set_kinetic_energy(self, energy):
        self.vel = scale_vec(self.vel, np.sqrt(2*energy/self.mass))

    def draw(self, surface):
        pos = (self.pos - np.array([self.radius/2, self.radius/2])).astype(int)
        surface.blit(self.image, pos)

    def in_bounds(self, xmin, ymin, xmax, ymax):
        if xmin < self.pos[0] < xmax and ymin < self.pos[1] < ymax:
            return True
        else:
            return False

    def wall_collision(self, w, dt):
        if w.active:
            next_pos = self.pos + self.vel*dt + scale_vec(self.vel, self.radius)
            ta, tb, intersection_point = intersection(self.pos[0], next_pos[0],
                                                      w.start[0], w.end[0],
                                                      self.pos[1], next_pos[1],
                                                      w.start[1], w.end[1])
            if 0 <= ta <= 1 and 0 <= tb <= 1:
                self.vel = self.vel - 2 * (np.dot(self.vel, w.normal)) * w.normal
                return True
            else:
                return False
        else:
            return False


class Wall:
    """
    A wall that used to contain the particles.
    It has a starting point, an end point, a width
    and a color.
    """
    def __init__(self, start, end, width=1, color=[255]*3):
        self.start = start
        self.end = end
        self.width = width
        self.color = color
        self.active = True

        # Additional wall properties
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        self.dir = normalize(np.array([dx, dy]))
        self.angle = np.arctan2(dy, dx)
        self.normal = rotate(self.dir, pi/2)
        self.length = np.linalg.norm(np.array([dx, dy]))
        self.vec = self.dir * self.length
        self.center = (self.start + self.end) / 2

    def set_pos(self, pos):
        self.start = pos - self.vec/2
        self.end = pos + self.vec/2

    def draw(self, surface):
        if self.active:
            pygame.draw.line(surface, self.color,
                             self.start.astype(int),
                             self.end.astype(int),
                             self.width)

    def wall_intersection(self, w2):
        return intersection(self.start[0], self.end[0],
                            w2.start[0], w2.end[0],
                            self.start[1], self.end[1],
                            w2.start[1], w2.end[1])


class Grid:
    """
    Used to improve computation time.
    Essentially implements neighbor lists.
    """
    def __init__(self,
                 Nx, Ny,
                 Sx, Sy,
                 Ex, Ey):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = (Ex-Sx)/Nx
        self.Ly = (Ey-Sy)/Ny

        self.reset()

    def reset(self):
        self.objects = [[[] for _ in range(self.Ny)]
                            for _ in range(self.Nx)]

    def draw(self, surface):
        for i, row in enumerate(self.objects):
            for j, cell in enumerate(row):
                pygame.draw.rect(surface, [50, 0, 50],
                                 (i*self.Lx+min_x, j*self.Ly+min_y,
                                  self.Lx, self.Ly),
                                 2)

    def add_object(self, p):
        # Determine cell indices
        cellx = int(np.floor((p.pos[0]-min_x)/self.Lx))
        celly = int(np.floor((p.pos[1]-min_y)/self.Ly))

        # Insert particle to the correct cell
        if (0 <= cellx < self.Nx) and (0 <= celly < self.Ny):
            self.objects[cellx][celly].append(p)
            p.set_cell(cellx, celly)


########################
# Simulation functions #
########################

def distribute_no_overlaps(particles,
                           min_x, min_y,
                           max_x, max_y):
    """
    Distributes particles such that they
    do not overlap each other.
    """
    # Initialize position of the first particle
    particles[0].pos = np.array([(max_x+min_x)/2,
                                 (max_y+min_y)/2])

    # iteratively place each successive particle
    # at a position that does not overlap with
    # any of the previous particles
    for i, p in enumerate(particles):
        overlap = True
        while overlap:
            overlap = False
            for p2 in particles[:i]:
                p.pos = np.random.uniform((min_x, min_y),
                                          (max_x, max_y),
                                          (1,2)).flatten()
                if dist(p.pos, p2.pos) <= (p.radius + p2.radius):
                    overlap = True


def particle_interaction(p1, p2, dt):
    """
    Deals with interaction between two particles.
    """
    dr = p2.pos - p1.pos # points from p1 to p2
    distance = np.linalg.norm(dr)
    if distance <= p1.radius + p2.radius:
        # If they collide - just Newtonian physics
        # (conservation of momentum and energy)
        dv = p2.vel - p1.vel
        dR = np.dot(dv, dr) / np.dot(dr, dr) * dr
        M = p1.mass + p2.mass
        overlap = p1.radius + p2.radius - distance
        if overlap > 0.0:
            p2.pos += normalize(dr) * overlap
        p1.vel += (2*p2.mass/M * dR)
        p2.vel -= (2*p1.mass/M * dR)


def vel_cm(objects):
    """
    Calculates the center of mass velocity
    for a list of particles. Used to make
    sure that the total momentum of the system
    is set to zero (otherwise we get a drift).
    """
    velx = np.sum([object.vel[0] for object in objects])
    vely = np.sum([object.vel[1] for object in objects])
    return np.array([velx, vely])


#####################
# Initialize pygame #
#####################

w, h = 1200, 1000
center = np.array([w/2, h/2])
pygame.display.init()
screen = pygame.display.set_mode((w, h))
pygame.font.init()
pygame.display.flip()


#################################
# Parameters for the simulation #
#################################

dt = 1.0
num_particles = 200

# Used for the containing box
min_x = 200
max_x = w - 600
min_y = 50
max_y = h - 50

# Gravity related parameters
G = 1.5E4
yref = max_y + 1000
GRAVITY = False

# Height histogram parameters
nbars = 30
bins = np.linspace(min_y, max_y, nbars)
sn = ((max_y - min_y)/nbars+1)
total_histogram = np.zeros(nbars-1)

# Display grid
show_grid = False

# Status text setup
gravity_status_text = TextOnScreen(pos=(10,0),
                                   color=(0, 250, 120),
                                   text='Gravity: ' + status_dict[GRAVITY])
grid_text = TextOnScreen(pos=(10,20),
                         color=(0, 250, 120),
                         text='')
startup_text = TextOnScreen(pos=(0,h//2),
                            color=(255, 255, 255),
                            size=50,
                            text='Initializing starting positions...',
                            centered=True)
startup_text.display(screen)
pygame.display.update()


##################
# Set up objects #
##################

load_params(argv[1])

#############
# Main loop #
#############

run = True
while run:
    ###############
    # Input event #
    ###############

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                run = False
            if event.key == pygame.K_w:
                walls[0].active = not walls[0].active
            if event.key == pygame.K_c:
                show_grid = not show_grid
            if event.key == pygame.K_g:
                GRAVITY = not GRAVITY
                gravity_status_text.set_text('Gravity: ' + status_dict[GRAVITY])
        if event.type == pygame.MOUSEBUTTONUP:
            pos = pygame.mouse.get_pos()
            for m in molecules:
                if dist(pos, m.pos) <= m.radius + 5:
                    m.flip_selection_status()


    ######################
    # Simulation physics #
    ######################

    # Place in grid
    grid.reset()
    for m in molecules:
        grid.add_object(m)

    # Create neighbor lists
    for m in molecules:
        m.set_neighors(grid)

    # Physics
    for b1 in molecules:
        if GRAVITY:
            b1.gravity(yref, dt)
        for wl in walls:
            b1.wall_collision(wl, dt)
        for b2 in b1.neighbors:
            particle_interaction(b1, b2, dt)

    # Move
    for m in molecules:
        m.move(dt)
        if not (min_x <= m.pos[0] <= max_x):
            m.pos[0] = np.random.uniform(min_x, max_x)
        if m.pos[1] >= max_y:
            molecules.remove(m)


    ###################
    # Data collection #
    ###################

    # Kinetic energy
    Ek = np.sum([m.get_kinetic_energy() for m in molecules])
    Ep = np.sum([m.get_potential_grav_energy(yref) for m in molecules])
    Et = Ek + Ep

    # Positions histogram
    ys = np.array([m.pos[1] for m in molecules])
    hist, _ = np.histogram(ys, bins)
    hist = hist / np.max(hist)
    if GRAVITY:
        total_histogram += hist


    #####################
    # Drawing to screen #
    #####################

    # Fill black
    screen.fill(3*[0])

    # Grid
    if show_grid:
        grid.draw(screen)

    # Balls
    for m in molecules:
        m.draw(screen)

    # Walls
    for wl in walls:
        wl.draw(screen)

    # Display histogram
    for i, hbar in enumerate(hist):
        bar = np.array([max_x+10, sn*(i+1.7), (w-max_x-100)*hbar, sn-3]).astype(int)
        pygame.draw.rect(screen, [10, 250, 100], bar)

    # Display info
    gravity_status_text.display(screen)
    grid_text.set_text('Grid: {}'.format(status_dict[show_grid]))
    grid_text.display(screen)

    # Black line between box and histogram
    pygame.draw.line(screen, [0]*3,
                     (max_x+6, min_y),
                     (max_x+6, max_y),
                     5)

    # Update screen
    pygame.display.update()


################
# Exit program #
################

pygame.quit()

with open('histogram.data', 'w') as f:
    for i, h in enumerate(total_histogram):
        f.write('{} {}\n'.format(i, h))

exit()
