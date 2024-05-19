import pygame
from pygame.locals import *
from actors import Agent, Wall, Landmark
from evolved_agent import EvolvedAgent
from env_evolution import PygameEvolvedEnviroment, EnvEvolution
from utils import intersection, distance_from_wall, angle_from_vector
from parameters import *
from nn import NN
from evolution import Evolution
from joblib import Parallel, delayed


# Parametri
"""
fitness_scores = np.zeros(AGENT_NUMBER)

agents = [
    EvolvedAgent(
        x=X_START,
        y=Y_START,
        n_sensors=N_SENSORS,
        controller=NN(N_SENSORS, activation=ACTIVATION),
        size=AGENT_SIZE,
        color=AGENT_COLOR,
        max_distance=MAX_DISTANCE,
    )
    for _ in range(AGENT_NUMBER)
]
envs = [
    EnvEvolution(
        agent, height=HEIGHT, width=WIDTH, instants=INSTANTS, w1=W1, w2=W2, w3=W3
    )
    for agent in agents
]

evl = Evolution(
    initial_population=agents,
    input_dim=INPUT_SIZE + 4,
    hidden_dim=32,
    layer_dim=4,
    output_dim=2,
)  # Passare alcuni parametri in input, che gestiscono l'evoluzione come la procedura di selezione, crossover e mutazione


"""
evl = Evolution(
    initial_population_size=AGENT_NUMBER,
    input_dim=INPUT_SIZE + 4,
    hidden_dim=32,
    layer_dim=4,
    output_dim=2,
)
evl.create_population()
envs = evl.population
time = 0
fitness_scores = np.zeros(AGENT_NUMBER)


for env in envs:
    env.load_landmarks(LANDMARK_TXT, LANDMARK_SIZE, LANDMARK_COLOR)
    env.load_walls(WALLS_TXT)

for generation in range(GENERATIONS):
    for env, index in zip(envs, range(AGENT_NUMBER)):
        env.reset()
        for _ in range(INSTANTS):
            env.move_agent()
        fitness_scores[index] = env.fitness_score()
    print(f"Generation {generation} - Fitness scores: {fitness_scores} simulating...")

    evl.rank_based_selection(fitness_scores)
    evl.mutation()
    evl.crossover()


"""

# initialize the pygame enviroment
pygame.init()
windows = [
    pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
    for _ in range(AGENT_NUMBER)
]

for env in envs:
    env.load_landmarks(LANDMARK_TXT, LANDMARK_SIZE, LANDMARK_COLOR)
    env.load_walls(WALLS_TXT)
clock = pygame.time.Clock()
pygame.display.set_caption(GAME_TITLE)


pause_state, show_text = False, False

start = None

while True:
    for generation in range(GENERATIONS):
        # Reset agents' positions at the start of each generation
        for env in envs:
            env.reset(random=True)

        if generation > 0:
            for i, env in enumerate(envs):
                fitness_scores[i] = env.fitness_score()
            print(f"Generation {generation} - Fitness scores: {fitness_scores}")
            evl.rank_based_selection(fitness_scores)
            evl.mutation()
            for env in envs:
                env.movements = 0

        if generation == GENERATIONS - 1:
            pygame.quit()
            quit()

        for _ in range(INSTANTS):
            for window in windows:
                window.fill(SCREEN_COLOR)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()
                    elif event.key == pygame.K_n:
                        if start is None:
                            start = pygame.mouse.get_pos()
                        else:
                            end = pygame.mouse.get_pos()
                            for env in envs:
                                env.add_wall(Wall(start[0], start[1], end[0], end[1]))
                            start = None
                    elif event.key == pygame.K_s:
                        for env in envs:
                            env.save_walls("./data/walls.txt")
                    elif event.key == pygame.K_BACKSPACE:
                        for env in envs:
                            env.walls.clear()
                    elif event.key == pygame.K_t:
                        show_text = not show_text
                    elif event.key == pygame.K_SPACE:
                        pause_state = not pause_state

            if start:
                for window in windows:
                    pygame.draw.line(window, "blue", start, pygame.mouse.get_pos(), 5)

            if not pause_state:
                for env, window in zip(envs, windows):
                    env.move_agent()
                    env.show(window)
                    # Uncomment the following line if you want to draw sensors
                    # env.draw_sensors(window, show_text=show_text)

            pygame.display.flip()
            clock.tick(FPS)
"""
