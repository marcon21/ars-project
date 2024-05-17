import pygame
from pygame.locals import *
from actors import Agent, Wall, Landmark
from evolved_agent import EvolvedAgent
from env_evolution import PygameEvolvedEnviroment, EnvEvolution
from utils import intersection, distance_from_wall, angle_from_vector
from parameters import *
from nn import NN
from evolution import Evolution


'''
    1. The evolution of the agents
    

    The evolution of the agents is done in the following way:
    
    1.1. Create the population of agents
    1.2. Create the enviroments
    1.3. Run the simulation for a number of generations
    1.4. For each generation:
        1.4.1. Load the landmarks and the walls in the enviroments
        1.4.2. Run the simulation for a number of instants
        1.4.3. Compute the fitness score of each agent
        1.5. Selection, mutation and crossover of the agents
        1.6. Update the population of agents
        
'''
#create population of agents
fitness_scores = np.zeros(AGENT_NUMBER)
agents = [EvolvedAgent(x=X_START,y = Y_START,n_sensors=N_SENSORS, controller=NN(N_SENSORS,activation=ACTIVATION),size=AGENT_SIZE,color=AGENT_COLOR, max_distance=MAX_DISTANCE) for _ in range(AGENT_NUMBER)]
envs = [EnvEvolution(agent, height=HEIGHT, width=WIDTH, instants=INSTANTS, w1=W1, w2=W2, w3=W3) for agent in agents]

#evolution
evl = Evolution(initial_population=agents) # io direi di passare alcuni parametri in input, che gestiscono l'evoluzione come la procedura di selezione, crossover e mutazione
for generation in range(GENERATIONS):
    for env in envs:
        env.load_landmarks(LANDMARK_TXT, LANDMARK_SIZE, LANDMARK_COLOR)
        env.load_walls(WALLS_TXT)
        for _ in range(INSTANTS):
            env.move_agent()
        fitness_scores[envs.index(env)] = env.fitness_score()
    print(f"Generation {generation} - Fitness scores: {fitness_scores}")
    
    #selection, mutation and crossover, update the population
    evl.selection(fitness_scores)
    evl.mutation()
    evl.crossover()
        
best_agents = evl.best_agent()

'''
    2. Show the best agents    
'''

pygame.init()
windows = [pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF) for _ in range(AGENT_NUMBER)]
envs = [PygameEvolvedEnviroment(agent, height=HEIGHT, width=WIDTH, instants=INSTANTS, w1=W1, w2=W2, w3=W3) for agent in best_agents]
for env in envs:
    env.load_landmarks(LANDMARK_TXT, LANDMARK_SIZE, LANDMARK_COLOR)
    env.load_walls(WALLS_TXT)
clock = pygame.time.Clock()
pygame.display.set_caption(GAME_TITLE)
frame_count = 0
pause_state, show_text = False, False

start = None

while True:
    windows[0].fill(SCREEN_COLOR)

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
        for env, window in zip(envs, windows):
            env.move_agent()
            env.draw_sensors(window, show_text=show_text)
            env.show(window)

    pygame.display.flip()

    clock.tick(FPS)  

