import pygame
from pygame.locals import *
from actors import Agent, Wall, Landmark
from evolved_agent import EvolvedAgent
from env_evolution import PygameEvolvedEnviroment
from utils import intersection, distance_from_wall, angle_from_vector
from parameters import *
from nn import NN

# Pygame setup
pygame.init()
window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
pygame.display.set_caption(GAME_TITLE)
dt = 1

# Inizializzazione dell'agente e dell'ambiente
agent = EvolvedAgent(n_sensors=12, controller=NN())
env = PygameEvolvedEnviroment(agent)
env.load_landmarks(LANDMARK_TXT, LANDMARK_SIZE, LANDMARK_COLOR)
env.load_walls(WALLS_TXT)

# Variabili di stato
pause_state, show_text = False, False
start = None

print(INSTRUCTIONS)

# Loop principale del gioco
while True:
    window.fill(SCREEN_COLOR)

    # Gestione degli eventi
    for event in pygame.event.get():
        if event.type == QUIT:
            quit()
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                quit()
            if event.key in (K_q, K_LEFT):
                agent.turn_direction -= ROTATION_SIZE
            if event.key in (K_e, K_RIGHT):
                agent.turn_direction += ROTATION_SIZE
            if event.key == K_n:
                start = (
                    pygame.mouse.get_pos()
                    if start is None
                    else (
                        env.add_wall(
                            Wall(
                                start[0],
                                start[1],
                                pygame.mouse.get_pos()[0],
                                pygame.mouse.get_pos()[1],
                            )
                        ),
                        None,
                    )[1]
                )
            if event.key == K_s:
                env.save_walls("./data/walls.txt")
            if event.key == K_BACKSPACE:
                env.walls.clear()
            if event.key == K_t:
                show_text = not show_text
            if event.key == K_SPACE:
                pause_state = not pause_state

    if start:
        pygame.draw.line(window, "blue", start, pygame.mouse.get_pos(), 5)
    
    # Movimento dell'agente
    if not pause_state:
        env.move_agent()
    
    # Disegno dell'ambiente e degli agenti
    env.draw_sensors(window, show_text=show_text)
    env.show(window)

    # Aggiornamento della finestra
    pygame.display.flip()

    dt = clock.tick(FPS) / 1000
