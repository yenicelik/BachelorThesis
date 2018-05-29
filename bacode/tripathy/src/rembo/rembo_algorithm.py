

class Rembo:

    def __init__(self):
        self.A = randommatrix
        self.remboDomain = ContinuousDomain(u, l)  # bound according to remobo paper
        self.optimizer = ScipyOptimizer(self.remboDomain)


    def next(self):
        x, _
        self.optimizer(self.u)  # define acquisition function u somehwere
        return x