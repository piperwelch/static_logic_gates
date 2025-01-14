from particle import Particle 


class Material: 
    def __init__(self, random, change_stiffness, change_size, id):
        #genome of a materials is array of particles, each of which has a stiffness and a inflation percent 
        self.change_stiffness = change_stiffness
        self.change_size = change_size
        self.num_particles = 10
        self.mutation_rate = 0.2
        self.id = id
        self.age = 0 
        self.fitness_map = {}
        self.make_particles(random)


    def make_particles(self, random): 
        self.particles = []
        for i in range(self.num_particles):
            stiffness = random.choice([1, 0.116279]) if self.change_stiffness else 1
            expansion = random.choice([0, 0.04, 0.02]) if self.change_size else 0
            self.particles.append(Particle(stiffness=stiffness, expansion=expansion))
        
        avg_stiff = 0 
        for particle in self.particles:
            avg_stiff += particle.stiffness
        self.average_stiffness = avg_stiff/10
    

    def mutate(self, random):
        for particle in self.particles:
            if self.change_size: #mutate expansion 
                if random.random() < self.mutation_rate:
                    old_expansion = particle.expansion.copy()
                    new_expansion = random.choice([0, 0.04, 0.02])

                    while new_expansion == old_expansion:
                        new_expansion = random.choice([0, 0.04, 0.02])
                    particle.expansion = new_expansion 


            if self.change_stiffness: #mutate stiffness 
                if random.random() < self.mutation_rate:
                    old_stiffness = particle.stiffness.copy()
                    new_stiffness = random.choice([1, 0.116279])

                    while new_stiffness == old_stiffness:
                        new_stiffness = random.choice([1, 0.116279])
                    particle.stiffness = new_stiffness


        avg_stiff = 0 
        for particle in self.particles:
            avg_stiff += particle.stiffness
        self.average_stiffness = avg_stiff/10
            

    def print(self, verbose):
        print(f'fitness: {self.fitness}, age: {self.age}, id: {self.id}')

        for particle in self.particles: 
            print("particle stiffness", particle.stiffness)
