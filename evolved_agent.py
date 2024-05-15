import actors

class EvolvedAgent(actors.Agent):
    def __init__(self, genome):
        self.genome = genome
        self.controller = self.genome.to_neural_network()

    def act(self, state):
        return self.brain.forward(state)

    def __str__(self):
        return f"EvolvedAgent(genome={self.genome})"
    


    