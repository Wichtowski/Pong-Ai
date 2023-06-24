import pygame
from random import randint

class Paddle:
    VEL = 20
    WIDTH = 20
    HEIGHT = 100

    def __init__(self, x, y):
        self.x = self.original_x = x
        self.y = self.original_y = y

    def draw(self, win):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
                
        pygame.draw.rect(
            win, (r, g, b), (self.x, self.y, self.WIDTH, self.HEIGHT))

    def move(self, up=True):
        if up:
            self.y -= self.VEL
        else:
            self.y += self.VEL

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
