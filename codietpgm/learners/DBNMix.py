class DBNMix:
    def __init__(self, dbn_instances):
        self.dbn_instances = dbn_instances

    def random_choice(self):
        # Return a randomly chosen DBN instance
        return np.random.choice(self.dbn_instances)
