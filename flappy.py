#! /usr/bin/env python

import pygame
import os
import random
import neat

pygame.font.init()

WIN_WIDTH = 600
WIN_HEIGHT = 800

STAT_FONT = pygame.font.SysFont("ubuntu", 50)

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join('textures',"pipe.png")))
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("textures","bg.png")), (600, 900))
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join('textures',"base.png")))
bird_img = pygame.transform.scale2x(pygame.image.load(os.path.join('textures',"bird.png")))

gen = 0

class Bird:
  global bird_img

  IMG = bird_img
  MAX_ROT = 25
  ROT_VEL = 20

  def __init__(self,x,y):

    self.x = x
    self.y = y
    self.tilt = 0
    self.vel = 0
    self.height = self.y
    self.img = self.IMG
    self.tick_count = 0

  def jump(self):

    self.vel = -10.5
    self.tick_count = 0
    self.height = self.y

  def move(self):

    self.tick_count += 1

    disp = (self.vel*(self.tick_count))+(1.5*((self.tick_count)**2))

    if disp >= 16:
      disp = (disp/abs(disp))*16 

    if disp < 0:
      disp -= 2
    
    self.y = self.y + disp

    if disp < 0 or self.y < self.height + 50:
      if self.tilt < self.MAX_ROT:
        self.tilt = self.MAX_ROT
    
    else:
      if self.tilt > -80:
        self.tilt -= self.ROT_VEL

  def draw(self,win):

    blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

  def get_mask(self):

    return pygame.mask.from_surface(self.img)

class Base:

  VEL = 5
  WIDTH = base_img.get_width()
  IMG = base_img

  def __init__(self,y):
    self.y = y
    self.x1 = 0
    self.x2 = self.WIDTH

  def move(self):

    self.x1 -= self.VEL
    self.x2 -= self.VEL

    if self.x1 + self.WIDTH < 0:
      self.x1 = self.x2 + self.WIDTH

    if self.x2 + self.WIDTH < 0:
      self.x2 = self.x1 + self.WIDTH

  def draw(self,win):

    win.blit(self.IMG,(self.x1,self.y))
    win.blit(self.IMG,(self.x2,self.y))

class Pipe():

  GAP = 200
  VEL = 5

  def __init__(self,x):
    self.x = x
    self.height = 0
    self.top = 0
    self.bottom = 0
    self.PIPE_TOP = pygame.transform.flip(pipe_img,False,True)
    self.PIPE_BOTTOM = pipe_img
    self.passed = False

    self.set_pipes()
    
  def set_pipes(self):

    self.height = random.randrange(50,450)
    self.top = self.height - self.PIPE_TOP.get_height()
    self.bottom = self.height + self.GAP

  def move(self):
    self.x -= self.VEL

  def draw(self,win):
    win.blit(self.PIPE_TOP, (self.x,self.top))
    win.blit(self.PIPE_BOTTOM, (self.x,self.bottom))

  def collide(self,bird):
    bird_mask = bird.get_mask()
    top_mask = pygame.mask.from_surface(self.PIPE_TOP)
    bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

    top_off = (self.x - bird.x, self.top - round(bird.y))
    botton_off = (self.x - bird.x, self.bottom - round(bird.y))

    b_point = bird_mask.overlap(bottom_mask, botton_off)
    t_point = bird_mask.overlap(top_mask, top_off)

    if t_point or b_point:
      return True

    return False

##======================================================##

def blitRotateCenter(surf, image, topleft, angle):
  rotated_image = pygame.transform.rotate(image, angle)
  new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)

  surf.blit(rotated_image, new_rect.topleft)

def draw_window(win, birds, base, pipes, score):
  global bg_img
  global gen
  
  win.blit(bg_img,(0,0))
  for bird in birds:
    bird.draw(win)
  base.draw(win)
  for pipe in pipes:
    pipe.draw(win)
  text = STAT_FONT.render("Score: " + str(score),1 , (255,255,255))
  win.blit(text,(WIN_WIDTH - text.get_width() -10 ,10))
  gen_text = STAT_FONT.render("Gen: " + str(gen-1),1,(255,255,255))
  win.blit(gen_text,(10,10))

  pygame.display.update()

##======================================================##

def main(genomes,config):

  global gen
  gen += 1
  birds = []
  nets = []
  ge = []

  for _,g in genomes:
    net = neat.nn.FeedForwardNetwork.create(g,config)
    nets.append(net)
    birds.append(Bird(230,350))
    g.fitness = 0
    ge.append(g)

  base = Base(730)
  pipes = [Pipe(600)]
  win = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
  clock = pygame.time.Clock()

  score = 0
 
  run =True
  while run and len(birds) > 0:
    clock.tick(30)

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        run = False
        pygame.quit()
        quit()

    pipe_ind = 0
    if len(birds) > 0:
      if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
        pipe_ind = 1

      

    for x, bird in enumerate(birds):
      bird.move()
      ge[x].fitness = 0.1

      output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

      if output[0] > 0.5:
        bird.jump()

    rem = []
    add_pipe = False
    for pipe in pipes:
      for x, bird in enumerate(birds):
        if pipe.collide(bird):
          ge[x].fitness -= 1
          birds.pop(x)
          nets.pop(x)
          ge.pop(x)

        if not pipe.passed and pipe.x < bird.x:
          pipe.passed = True
          add_pipe = True

      if pipe.x + pipe.PIPE_TOP.get_width() < 0:
        rem.append(pipe)
      pipe.move()
      
    if add_pipe:
      score += 1
      for g in ge:
        g.fitness += 5
      pipes.append(Pipe(600))

    for r in rem:
      pipes.remove(r)    

    for x,bird in enumerate(birds):
      if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
        birds.pop(x)
        nets.pop(x)
        ge.pop(x)

    base.move()
    draw_window(win,birds,base,pipes,score)



def run(config_path):
  config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

  pop = neat.Population(config)
  pop.add_reporter(neat.StdOutReporter(True))
  stats = neat.StatisticsReporter()
  pop.add_reporter(stats)

  winner = pop.run(main,50)

  print("\n Best Genome: \n {!s}".format(winner))


if __name__ == "__main__":

  local_dir = os.path.dirname(__file__)
  config_path = os.path.join(local_dir, "config-feedforward.txt")
  run(config_path)
