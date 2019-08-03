#!/usr/bin/env python3
# -*- coding: iso-8859-15 -*-

import sys
import numpy as np
from numpy import pi, sin, cos, sqrt
import pygame


# For displaying data on screen
status_dict = {True: 'ON',
               False: 'OFF'}


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
    returns the intersecion
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
    returns the Eucledian distance
    between two points p1, p2
    """
    return np.linalg.norm(p2-p1)


class TextOnScreen:
    """
    A class to display a text
    on screen, at a chosen location,
    font, size, etc.
    """
    def __init__(self, pos=(0,0), color=(0, 200, 0),
                       font='Cabin', size=15, text=''):
        self.pos = pos
        self.color = color
        self.font = pygame.font.SysFont(font, size)
        self.text = text

    def set_text(self, text):
        self.text = text

    def display(self, surface):
        render = self.font.render(self.text,
                                  False,
                                  self.color)
        surface.blit(render, self.pos)


###########
# Classes #
###########

class Particle:
    """
    A particle with position, velocity, mass
    and a radius. Moves according to Newton's
    laws of motion, including GRAVITY.
    It also exist in a cell, to reduce
    computational costs (i.e. "neighbor lists")
    """
    def __init__(self, pos, vel,
                 mass, radius,
                 Gravity=False,
                 color=[255, 255, 255]):

        if pos == 'random':
            self.pos = np.random.uniform(low  = (min_x, min_y+400),
                                         high = (max_x, max_y),
                                         size = (1,2)).flatten()
        else:
            self.pos = pos

        self.vel = vel
        self.speed = np.linalg.norm(self.vel)
        self.mass = mass
        self.radius = radius
        self.Gravity = Gravity

        self.color = color
        self.old_color = self.color
        self.selected = False

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
        self.old_color = self.color
        self.color = [255, 255, 255]

    def unselect(self):
        self.selected = False
        self.color = self.old_color

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
        pygame.draw.circle(surface, self.color,
                           self.pos.astype(int),
                           self.radius)
        if self.selected:
            vel_vec = self.vel * 40
            pygame.draw.line(surface, [255, 0, 0], self.pos, (self.pos+vel_vec).astype(int), 4)

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
                pygame.draw.rect(surface, [100, 100, 100],
                                 (i*self.Lx+min_x, j*self.Ly+min_y,
                                  self.Lx, self.Ly),
                                 3)

    def add_object(self, p):
        # Determine cell indeces
        cellx = int(np.floor((p.pos[0]-min_x)/self.Lx))
        celly = int(np.floor((p.pos[1]-min_y)/self.Ly))

        # Insert particle to the correct cell
        if (0 <= cellx < self.Nx) and (0 <= celly < self.Ny):
            self.objects[cellx][celly].append(p)
            p.set_cell(cellx, celly)


########################
# Simulation functions #
########################

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
    #elif LJ:
    #    # Lennard-Jones potential
    #    F12 = LJ_coeff * 12 / distance**13 * p1.radius**6 * (distance**6 - p1.radius**6)
    #    F21 = LJ_coeff * 12 / distance**13 * p2.radius**6 * (distance**6 - p2.radius**6)
    #    p1.add_acceleration(scale_vec(+dr, F12/p1.mass), dt)
    #    p2.add_acceleration(scale_vec(-dr, F21/p2.mass), dt)


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
min_y = 0
max_y = h

# Gravity related parameters
G = 1.5E4
yref = max_y + 1000
GRAVITY = False

# Height histogram parameters
nbars = 30
bins = np.linspace(min_y, max_y, nbars)
sn = (max_y/nbars+1)

# Display grid
show_grid = False

# Status text setup
gravity_status_text =  TextOnScreen(pos=(10,0),
                                    color=(0, 250, 120),
                                    text='Gravity: ' + status_dict[GRAVITY])
ke_text =  TextOnScreen(pos=(10,20),
                        color=(0, 250, 120),
                        text='')
pe_text =  TextOnScreen(pos=(10,40),
                        color=(0, 250, 120),
                        text='')
te_text =  TextOnScreen(pos=(10,60),
                        color=(0, 250, 120),
                        text='')
grid_text =  TextOnScreen(pos=(10,80),
                          color=(0, 250, 120),
                          text='')


##################
# Set up objects #
##################

# Particles
balls = [Particle(pos='random',
                  vel=np.random.uniform(-1, 1, 2),
                  mass=1,
                  radius=10,
                  Gravity=True,
                  color=[100, 200, 255])
         for _ in range(num_particles)]

# Set initial kinetic energy
Ek = np.sum([b.get_kinetic_energy() for b in balls])
for b in balls:
    b.set_kinetic_energy(500/num_particles)

# Walls
walls = []
walls.append(Wall(start=np.array([min_x, min_y]),
                  end=np.array([max_x, min_y]),
                  width=4,
                  color=[255,255,255]))
walls.append(Wall(start=np.array([min_x, min_y]),
                  end=np.array([min_x, max_y]),
                  width=4,
                  color=[255,255,255]))
walls.append(Wall(start=np.array([max_x, max_y]),
                  end=np.array([min_x, max_y]),
                  width=4,
                  color=[255,255,255]))
walls.append(Wall(start=np.array([max_x, max_y]),
                  end=np.array([max_x, min_y]),
                  width=4,
                  color=[255,255,255]))

# Grid
grid = Grid(10, 20,
            min_x, min_y,
            max_x, max_y)


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
                gravity_status_text.set_text('gravity: ' + status_dict[GRAVITY])
        if event.type == pygame.MOUSEBUTTONUP:
            pos = pygame.mouse.get_pos()
            for b in balls:
                b.unselect()
                if dist(pos, b.pos) < b.radius:
                    b.select()


    ######################
    # Simulation physics #
    ######################

    # Place in grid
    grid.reset()
    for b in balls:
        grid.add_object(b)

    # Create neighbor lists
    for b in balls:
        b.set_neighors(grid)

    # Physics
    for b1 in balls:
        if GRAVITY:
            b1.gravity(yref, dt)
        for wl in walls:
            b1.wall_collision(wl, dt)
        for b2 in b1.neighbors:
            particle_interaction(b1, b2, dt)

    # Move
    for b in balls:
        b.move(dt)
        if not (min_x <= b.pos[0] <= max_x):
            b.pos[0] = np.random.uniform(min_x, max_x)
        if b.pos[1] <= min_y:
            balls.remove(b)


    ###################
    # Data collection #
    ###################

    # Kinetic energy
    Ek = np.sum([b.get_kinetic_energy() for b in balls])
    Ep = np.sum([b.get_potential_grav_energy(yref) for b in balls])
    Et = Ek + Ep

    # Positions histogram
    ys = np.array([b.pos[1] for b in balls])
    hist, _ = np.histogram(ys, bins)
    hist = hist / np.max(hist)


    #####################
    # Drawing to screen #
    #####################

    # Fill black
    screen.fill(3*[0])

    # Grid
    if show_grid:
        grid.draw(screen)

    # Balls
    for b in balls:
        b.draw(screen)

    # Walls
    for wl in walls:
        wl.draw(screen)

    # Display histogram
    for i, hbar in enumerate(hist):
        bar = np.array([max_x+10, sn*i, (w-max_x)*hbar, sn-3]).astype(int)
        pygame.draw.rect(screen, [10, 250, 100], bar)

    # Display info
    gravity_status_text.display(screen)
    ke_text.set_text('Kinetic energy = {:0.2f}'.format(Ek))
    pe_text.set_text('Potential energy = {:0.2f}'.format(Ep))
    te_text.set_text('Total energy = {:0.2f}'.format(Et))
    grid_text.set_text('Grid: {}'.format(status_dict[show_grid]))
    ke_text.display(screen)
    pe_text.display(screen)
    te_text.display(screen)
    grid_text.display(screen)

    # Update screen
    pygame.display.update()


################
# Exit program #
################

pygame.quit()
sys.exit()
