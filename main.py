import pygame
from pygame.locals import *
from actors import Agent, Wall, Landmark
from evolved_agent import EvolvedAgent
from env_evolution import PygameEvolvedEnviroment
from utils import intersection, distance_from_wall, angle_from_vector
from parameters import *
from nn import NN

pygame.init()
window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
pygame.display.set_caption(GAME_TITLE)

agent = EvolvedAgent(x=X_START,y = Y_START,n_sensors=12, controller=NN(),size=AGENT_SIZE,color=AGENT_COLOR, max_distance=MAX_RANGE)
env = PygameEvolvedEnviroment(agent, height=HEIGHT, width=WIDTH, instants=INSTANTS, w1=W1, w2=W2, w3=W3)
env.load_landmarks(LANDMARK_TXT, LANDMARK_SIZE, LANDMARK_COLOR)
env.load_walls(WALLS_TXT)
frame_count = 0
pause_state, show_text = False, False

start = None

while True:
    window.fill(SCREEN_COLOR)

    for event in pygame.event.get():
        if event.type in (QUIT, KEYDOWN) and event.key == K_ESCAPE:
            quit()
        if event.type == KEYDOWN:
            if event.key == K_n:
                start = (
                    pygame.mouse.get_pos() if start is None else (
                        env.add_wall(Wall(start[0], start[1], pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1])),
                        None
                    )[1]
                )
            elif event.key == K_s:
                env.save_walls("./data/walls.txt")
            elif event.key == K_BACKSPACE:
                env.walls.clear()
            elif event.key == K_t:
                show_text = not show_text
            elif event.key == K_SPACE:
                pause_state = not pause_state

    if start:
        pygame.draw.line(window, "blue", start, pygame.mouse.get_pos(), 5)
    

    
    frame_count += 1
    if frame_count == INSTANTS:
        quit()
    if not pause_state:
        env.move_agent()
    env.draw_sensors(window, show_text=show_text)
    env.show(window)

    pygame.display.flip()

    clock.tick(FPS)  

