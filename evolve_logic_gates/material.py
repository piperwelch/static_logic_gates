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
            # stiffness = random.choice([1,3]) if self.change_stiffness else 10
            stiffness = random.randint(1,10) if self.change_stiffness else 10

            expansion = random.uniform(0, 0.040) if self.change_size else 0
            # expansion = round(expansion, 3)
            # if stiffness > 5: stiffness = 2 
            # else: stiffness = 1 
            self.particles.append(Particle(stiffness=stiffness, expansion=expansion))
        avg_stiff = 0 
        for particle in self.particles:
            avg_stiff += particle.stiffness
        self.average_stiffness = avg_stiff/10
    

    def mutate(self, random):
        for particle in self.particles:
            if self.change_size: #mutate expansion 
                if random.random() < self.mutation_rate:
                    particle.expansion += random.uniform(-0.005, 0.005)
                    if particle.expansion > 0.040: particle.expansion = 0.040
                    if particle.expansion < 0.0: particle.expansion = 0.0
            # particle.expansion = round(particle.expansion, 3)

            if self.change_stiffness: #mutate stiffness 
                if random.random() < self.mutation_rate:
                    particle.stiffness += random.uniform(-0.5, 0.5)
                    # if particle.stiffness == 1:particle.stiffness = 3
                    # if particle.stiffness == 3:particle.stiffness = 1 

                    if particle.stiffness > 10: particle.stiffness = 10 
                    if particle.stiffness < 0.5: particle.stiffness = 0.5


        avg_stiff = 0 
        for particle in self.particles:
            avg_stiff += particle.stiffness
        self.average_stiffness = avg_stiff/10
        
            

    def print(self, verbose):
        print(f'fitness: {self.fitness}, age: {self.age}, id: {self.id}')

        for particle in self.particles: 
            print("particle stiffness", particle.stiffness)
