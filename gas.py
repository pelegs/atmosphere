#!/usr/bin/env python3
# -*- coding: iso-8859-15 -*-

import sys
import numpy as np
from numpy import pi, sin, cos, sqrt
import pygame


status_dict = {True: 'ON',
               False: 'OFF'}


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
            self.pos = np.random.uniform(low  = (min_x, min_y),
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


    def add_acceleration(self, a, dt):
        self.vel += a*dt

    def move(self, dt):
        self.pos += self.vel * dt
        self.speed = np.linalg.norm(self.vel)

    def get_kinetic_energy(self):
        return 0.5*self.mass*np.linalg.norm(self.vel)**2

    def set_kinetic_energy(self, energy):
        self.vel = scale_vec(self.vel, np.sqrt(2*energy/self.mass))

    def draw(self, surface):
        pygame.draw.circle(surface, self.color,
                           self.pos.astype(int),
                           self.radius)

    def in_bounds(self, xmin, ymin, xmax, ymax):
        if xmin < self.pos[0] < xmax and ymin < self.pos[1] < ymax:
            return True
        else:
            return False

    def wall_collision(self, w, dt):
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


class Wall:
    def __init__(self, start, end, width=1, color=[255]*3):
        self.start = start
        self.end = end
        self.width = width
        self.color = color

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
        self.objects = [[[] for _ in range(self.Nx)]
                            for _ in range(self.Ny)]

    def add_object(self, p):
        cellx = int(np.floor(p.pos[0]/self.Lx))
        celly = int(np.floor(p.pos[1]/self.Ly))
        if (0 <= cellx < self.Nx) and (0 <= celly < self.Ny):
            self.objects[cellx][celly].append(p)
            p.set_cell(cellx, celly)


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
    elif LJ:
        # Lennard-Jones potential
        F12 = LJ_coeff * 12 / distance**13 * p1.radius**6 * (distance**6 - p1.radius**6)
        F21 = LJ_coeff * 12 / distance**13 * p2.radius**6 * (distance**6 - p2.radius**6)
        p1.add_acceleration(scale_vec(+dr, F12/p1.mass), dt)
        p2.add_acceleration(scale_vec(-dr, F21/p2.mass), dt)


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


dt = 1.0
num_particles = 200
w, h = 1200, 1000
g = 0.1
GRAVITY = np.array([0, g])
G = False
LJ_coeff = 2E1
LJ = False
min_x = 0
max_x = w - 700
min_y = 0
max_y = h
balls = [Particle(pos='random',
                  vel=np.random.uniform(-1, 1, 2),
                  mass=1,
                  radius=10,
                  Gravity=True,
                  color=[255, 0, 0])
         for _ in range(num_particles)]

Ek = np.sum([b.get_kinetic_energy() for b in balls])
for b in balls:
    b.set_kinetic_energy(10/num_particles)

w1 = Wall(start=np.array([min_x, min_y]),
          end=np.array([max_x, min_y]),
          width=4,
          color=[255,255,255])
w2 = Wall(start=np.array([min_x, min_y]),
          end=np.array([min_x, max_y]),
          width=4,
          color=[255,255,255])
w3 = Wall(start=np.array([max_x, max_y]),
          end=np.array([min_x, max_y]),
          width=4,
          color=[255,255,255])
w4 = Wall(start=np.array([max_x, max_y]),
          end=np.array([max_x, min_y]),
          width=4,
          color=[255,255,255])
walls = [w1, w2, w3, w4]

grid = Grid(50, 50,
            min_x, min_y,
            max_x, max_y)

nbars = 20
sn = max_y/nbars

center = np.array([w/2, h/2])
pygame.display.init()
screen = pygame.display.set_mode((w, h))
pygame.font.init()
pygame.display.flip()

gravity_status_text =  TextOnScreen(pos=(10,0),
                                    color=(0, 250, 120),
                                    text='Gravity: ' + status_dict[G])
LJ_status_text =  TextOnScreen(pos=(10,20),
                               color=(0, 250, 120),
                               text='L-J: ' + status_dict[LJ])
ke_text =  TextOnScreen(pos=(10,40),
                        color=(0, 250, 120),
                        text='')
pe_text =  TextOnScreen(pos=(10,60),
                        color=(0, 250, 120),
                        text='')
te_text =  TextOnScreen(pos=(10,80),
                        color=(0, 250, 120),
                        text='')

# Main loop
run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                run = False
            if event.key == pygame.K_g:
                G = not G
                gravity_status_text.set_text('gravity: ' + status_dict[G])
            if event.key == pygame.K_l:
                LJ = not LJ
                LJ_status_text.set_text('L-J: ' + status_dict[LJ])

    # Place in grid
    grid.reset()
    for b in balls:
        grid.add_object(b)

    # Create neighbor lists
    for b in balls:
        b.set_neighors(grid)

    # Physics
    for b1 in balls:
        if G:
            b1.add_acceleration(GRAVITY, dt)
        for wl in walls:
            b1.wall_collision(wl, dt)
        for b2 in b1.neighbors:
            particle_interaction(b1, b2, dt)

    # Move
    for b in balls:
        b.move(dt)
        if not b.in_bounds(min_x, min_y,
                           max_x, max_y):
            # TODO: Put in such height h that
            # total energy is conserved
            balls.remove(b)

    # Velocities (for coloring)
    #vels = np.array([b.speed for b in balls])
    #min_vel = np.min(vels)
    #max_vel = np.max(vels)

    # Collect data
    # Kinetic energy
    Ek = np.sum([b.get_kinetic_energy() for b in balls])
    ke_text.set_text('Kinetic energy = {:0.2f}'.format(Ek))
    Ep = np.sum([b.mass * g * (max_y - b.pos[1]) for b in balls])
    pe_text.set_text('Potential energy = {:0.2f}'.format(Ep))
    Et = Ek + Ep
    te_text.set_text('Total energy = {:0.2f}'.format(Et))

    # Positions histogram
    ys = np.array([b.pos[1] for b in balls])
    hist, _ = np.histogram(ys, nbars)
    hist = hist / np.max(hist)

    """
    Drawing
    """
    # Fill black
    screen.fill(3*[0])

    # Balls
    for b in balls:
        b.draw(screen)

    # Walls
    for wl in walls:
        wl.draw(screen)

    # Display histogram
    for i, hbar in enumerate(hist):
        bar = np.array([max_x, sn*i, (w-max_x)*hbar, sn]).astype(int)
        pygame.draw.rect(screen, [0, 255, 0], bar)

    # Display info
    gravity_status_text.display(screen)
    LJ_status_text.display(screen)
    ke_text.display(screen)
    pe_text.display(screen)
    te_text.display(screen)

    # Update screen
    pygame.display.update()

# Exit program
pygame.quit()
sys.exit()
