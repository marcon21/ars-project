import pygame
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import pygame_widgets.widget
from pygame.locals import *
from actors import Agent, Wall, Landmark
from env import PygameEnviroment, Enviroment
from kalman_filter import Kalman_Filter, PygameKF
from utils import intersection, distance_from_wall, angle_from_vector
from math import pi, degrees,atan2
import numpy as np
from random import randint
from random import random as rand
from experiment1 import *

def reset_agent():
    agent.pos = (window.get_width() / 2, window.get_height() / 2)
    agent.direction_vector = np.array([1, 0])

# Pygame setup
pygame.init()
window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
dt = 0
pygame.display.set_caption(GAME_TITLE)

with open("output.txt", "w") as file:
    file.write("")


#SLIDERS TO CONTROL PARAMETERS 
slider = Slider(window, WIDTH- 230, 50, 200, 10, min=0, max=99, step=1)
output = TextBox(window, WIDTH- 230, 20, 30, 30, fontSize=17)
output.disable() 

slider_Rsx = Slider(window, WIDTH- 230, 50+ 80, 200, 10, min=0, max=100, step=0.05)
output1 = TextBox(window, WIDTH- 230, 20+80, 30, 30, fontSize=17)
output1.disable() 

slider_Rsy = Slider(window, WIDTH- 230, 50+ 160, 200, 10, min=0, max=100, step=0.05)
output2 = TextBox(window, WIDTH- 230, 20+160, 30, 30, fontSize=17)
output2.disable() 

slider_Rsth = Slider(window, WIDTH- 230, 50+ 240, 200, 10, min=0, max=10, step=0.05)
output3 = TextBox(window, WIDTH- 230, 20+240, 30, 30, fontSize=17)
output3.disable() 

slider_Qsx = Slider(window, WIDTH- 230, 50+ 320, 200, 10, min=0.05, max=100, step=0.05)
output4 = TextBox(window, WIDTH- 230, 20+320, 30, 30, fontSize=17)
output4.disable()

slider_Qsy = Slider(window, WIDTH- 230, 50+ 400, 200, 10, min=0.05, max=100, step=0.05)
output5 = TextBox(window, WIDTH- 230, 20+400, 30, 30, fontSize=17)
output5.disable()

slider_Qsth = Slider(window, WIDTH- 230, 50+ 480, 200, 10, min=0.05, max=10, step=0.05)
output6 = TextBox(window, WIDTH- 230, 20+480, 30, 30, fontSize=17)
output6.disable()

slider_range = Slider(window, WIDTH- 230, 50+ 560, 200, 10, min=0, max=300, step=1)
output7 = TextBox(window, WIDTH- 230, 20+560, 30, 30, fontSize=17)
output7.disable()




# Initialize agent
agent = Agent(x=X_START, y=Y_START, size=AGENT_SIZE,
              move_speed=BASE_MOVE_SPEED, n_sensors=SENSORS, max_distance=RANGE, color=AGENT_COLOR)

# Initialize environment and load landmarks
env = PygameEnviroment(agent=agent)
env.load_landmarks(LANDMARK_TXT, LANDMARK_SIZE, LANDMARK_COLOR)
env.load_walls(WALLS_TXT)


# Initialize Kalman Filter
kfr = PygameKF(env, MEAN, COV_MATRIX, R, Q)

# Set initial state
pause_state, show_text = False, False
start, end = None, None

print(INSTRUCTIONS)

def draw_legend():
    font = pygame.font.Font(None, 24)
    x, y = agent.pos
    theta = atan2(agent.direction_vector[1], agent.direction_vector[0])

    legend_text = [
    f" FPS = {FPS}",
    f"Position = [ x = {round(x)}, y = {round(y)}, theta = {round(degrees(theta))}]",
    f"Estimated pose = [ x = {round(kfr.mean[0])}, y = {round(kfr.mean[1])}, theta = {round(degrees(kfr.mean[2]))}]",
    f"Actual difference = [ x = {round(x - kfr.mean[0])}, y = {round(y - kfr.mean[1])}, theta = {round(degrees(theta - kfr.mean[2]))}]",
    f" Sensors = {agent.n_sensors}  Press s to add 2",
    "Sliders: [1]: FPS [2,3,4]: R  [5,6,7]: Q [8]: sensor range"
    
]
    y = 50 
    for text in legend_text:
        text_surface = font.render(text, True, "red")
        text_rect = text_surface.get_rect()
        text_rect.topleft = (50, y)
        window.blit(text_surface, text_rect)
        y += 30  


while True:
    
    with open("output.txt", "a") as file:
        window.fill(SCREEN_COLOR)
        
        
        while pause_state:
            for event in pygame.event.get():
                if event.type==KEYDOWN:
                    if event.key==K_SPACE:
                        pause_state = False
        events = pygame.event.get()
        for event in events:
            if event.type == QUIT: quit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE: quit()
                if event.key in (K_q, K_LEFT): agent.turn_direction -= ROTATION_SIZE
                if event.key == K_v : FPS+= 1
                if event.key == K_c : FPS-= 1
                if event.key in (K_e, K_RIGHT): agent.turn_direction += ROTATION_SIZE
                if event.key == K_n: start = pygame.mouse.get_pos() if start is None else (env.add_wall(Wall(start[0], start[1], pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1])), None)[1]
                if event.key == K_s: env.save_walls("walls.txt")
                if event.key == K_l: env.load_walls("walls.txt"), reset_agent()
                if event.key == K_BACKSPACE: env.walls.clear()
                if event.key == K_r: reset_agent()
                if event.key == K_t: show_text = not show_text
                if event.key == K_s: agent.n_sensors += 2
                if event.key == K_SPACE: pause_state = True
        
        #update variables based on sliders
        FPS = slider.getValue()  
        R[0][0] = slider_Rsx.getValue()
        R[1][1] = slider_Rsy.getValue()
        R[2][2] = slider_Rsth.getValue()
        Q[0][0] = slider_Qsx.getValue()
        Q[1][1] = slider_Qsy.getValue()
        Q[2][2] = slider_Qsth.getValue()
        agent.max_distance = slider_range.getValue()
                            
                            
        if start: pygame.draw.line(window, "blue", start, pygame.mouse.get_pos(), 5),
        env.agent.move_speed = BASE_MOVE_SPEED * dt * 0.5 * (not pause_state)
        if not pause_state: env.move_agent()
        env.draw_sensors(window, show_text=show_text), env.show(window), kfr.correction(), kfr.show(window),draw_legend(),
        
        #print output of sliders
        output.setText(slider.getValue()),output1.setText(slider_Rsx.getValue()),
        output2.setText(slider_Rsy.getValue()),output3.setText(slider_Rsth.getValue()),
        output4.setText(slider_Qsx.getValue()), output5.setText(slider_Qsy.getValue()), 
        output6.setText(slider_Qsth.getValue()), output7.setText(slider_range.getValue())
        pygame_widgets.update(events), 
        
        
        
        # log file of poses and estimations
        file.write(f"R = {kfr.R}, Q = {kfr.Q}")
        x,y = env.agent.pos
        theta = angle_from_vector(env.agent.direction_vector)
        file.write(f"{[x,y,theta]}\n")
        file.write(f"{kfr.mean}\n")
        
        
        
        #update window
        pygame.display.flip()
        
        dt = clock.tick(FPS) / 1000
