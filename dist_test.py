#!/usr/bin/env python3
# -*- coding: iso-8859-15 -*-

import numpy as np
import pygame
from numpy import pi, sin, cos, sqrt
import sys


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

def cross(v1, v2):
    return (v1[0]*v2[1] - v1[1]*v2[0])

def distance_point_wall(p, wall):
    AP = p - wall.start
    u = wall.dir
    return cross(AP, u)


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

    def draw(self, surface):
        if self.active:
            pygame.draw.line(surface, self.color,
                             self.start.astype(int),
                             self.end.astype(int),
                             self.width)


wall = Wall(np.array([300, 200]),
            np.array([600, 100]),
            1, (255, 255, 255))

w, h = 1200, 1000
center = np.array([w/2, h/2])
pygame.display.init()
screen = pygame.display.set_mode((w, h))
pygame.display.flip()

run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                run = False

    pos = pygame.mouse.get_pos()
    print(distance_point_wall(pos, wall))

    screen.fill(3*[0])
    wall.draw(screen)
    pygame.draw.circle(screen, (0, 255, 150), pos, 2)
    pygame.display.update()
