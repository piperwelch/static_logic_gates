'''
Created on 2024-06-20 10:40:59
@author: piperwelch 

Description: Implementation of Age-Fitness Pareto Optimization (AFPO) 
https://dl.acm.org/doi/10.1145/1830483.1830584

NOTE: For AFPO, population size must be significantly large so that not all individuals in the population are on the pareto front. 
'''

import numpy as np
import copy
import operator
import pickle
import os
import random
from material import Material 
import sys
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PlotFunctions import ConfigPlot_DiffSize
from concurrent.futures import ProcessPoolExecutor
from MD_functions import FIRE_VL, FIRE_FixedTopForce_VL, ForceWall
from PlotFunctions import ConfigPlot_DiffSize


class AFPO:

    def __init__(self, random_seed, gens, pop_size, \
    change_stiffness, change_size, target_particle, 
    checkpoint_every=50, gate="AND", encoding='size'):
        self.gate = gate
        self.seed = random_seed
        self.change_stiffness = change_stiffness
        self.change_size = change_size
        self.target_particle = target_particle
        # Seed rng 
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.gens = gens
        self.pop_size = pop_size
        self.checkpoint_every = checkpoint_every
        self.next_id = 0
        self.fitness_data = np.zeros(shape=(self.gens+1, self.pop_size, 3))
        self.encoding = encoding 
        os.makedirs('checkpoints/', exist_ok=True)

        self.create_initial_population()

    
    def create_initial_population(self):
        self.pop = [] # population is a list of materials
        for i in range(self.pop_size): # initialize random materials
            self.pop.append(Material(random= np.random, change_stiffness= self.change_stiffness, change_size = self.change_size, id = self.next_id))
            #TODO make materials
            self.next_id+=1 


    def run(self, continue_from_checkpoint=False, additional_gens=0):
        if continue_from_checkpoint:
            max_gens = self.curr_gen + additional_gens

            # Expand fitness data matrix to account for additional gens
            new_fitness_data = np.zeros((max_gens + 1, self.pop_size, 3))

            new_fitness_data[0:self.curr_gen,:,:] = self.fitness_data[0:self.curr_gen,:,:]
            self.fitness_data = new_fitness_data

            for i,material in enumerate(self.pop):   
                self.fitness_data[self.curr_gen,i,0] = material.fitness
                self.fitness_data[self.curr_gen,i,1] = material.age
                self.fitness_data[self.curr_gen,i,1] = material.average_stiffness
                
            self.curr_gen += 1

            self.gens = max_gens
            while self.curr_gen < self.gens + 1:

                self.perform_one_generation()
                if self.curr_gen % self.checkpoint_every == 0:
                    self.save_checkpoint()

                print("GEN: {}".format(self.curr_gen))
                if self.curr_gen % 5 == 0:
                    self.print_best(verbose=False)
                
                self.curr_gen += 1
        else:
            self.curr_gen = 0
            self.evaluate_generation_zero()
                        
            while self.curr_gen < self.gens + 1: # Evolutionary loop
                
                self.perform_one_generation()
                if self.curr_gen % self.checkpoint_every == 0:
                    self.save_checkpoint()
                
                print("GEN: {}".format(self.curr_gen), flush=True)
                if self.curr_gen % 50 == 0:
                    self.print_best(verbose=False)

                self.curr_gen += 1

        return self.return_best(), self.fitness_data


    def evaluate_generation_zero(self):
        
        # Evaluate individuals in the population
        
        self.evaluate(self.pop)

        for i,material in enumerate(self.pop):            
            # Record fitness statistics    
            self.fitness_data[self.curr_gen,i,0] = material.fitness
            self.fitness_data[self.curr_gen,i,1] = material.age
            self.fitness_data[self.curr_gen,i,2] = material.average_stiffness
            
        print("GEN: {}".format(self.curr_gen))
        self.print_best(verbose=False)

        self.curr_gen += 1


    def perform_one_generation(self):

        self.increase_age()
        children = self.breed()
        children = self.insert_random(children)

        # Evaluate children
        self.evaluate(children)
        for child_material in children:
            # Extend population by adding child material (extends to pop_size*2+1 individuals every generation then gets reduced back to pop_size)
            self.pop.append(child_material) 

        self.survivor_selection()

        # Record statistics 
        for i, material in enumerate(self.pop):
            self.fitness_data[self.curr_gen,i, 0] = material.fitness
            self.fitness_data[self.curr_gen,i, 1] = material.age
            self.fitness_data[self.curr_gen,i, 2] = material.average_stiffness
            

    def increase_age(self):
        for material in self.pop:
            material.age += 1


    def breed(self):
        children = []
        for i in range(self.pop_size):

            #cParent Selection via Tournament Selection (based on fitness only)
            parent = self.tournament_selection()
            
            # # Create offspring via mutation
            child = copy.deepcopy(self.pop[parent])
            child.id = self.next_id
            self.next_id += 1
            child.mutate(np.random)
            children.append(child)

        return children


    def insert_random(self, children):
        children.append(Material(random=np.random, change_stiffness=self.change_stiffness, change_size=self.change_size, id=self.next_id))
        self.next_id += 1
        return children


    def tournament_selection(self):
        p1 = np.random.randint(len(self.pop))
        p2 = np.random.randint(len(self.pop))
        while p1 == p2:
            p2 = np.random.randint(len(self.pop))

        if self.pop[p1].fitness > self.pop[p2].fitness:
            return p1
        else:
            return p2


    def survivor_selection(self):
        # Remove dominated individuals until the target population size is reached
        while len(self.pop) > self.pop_size:

            # Choose two different individuals from the population
            ind1 = np.random.randint(len(self.pop))
            ind2 = np.random.randint(len(self.pop))
            while ind1 == ind2:
                ind2 = np.random.randint(len(self.pop))

            if self.dominates(ind1, ind2):  # ind1 dominates
                
                # remove ind2 from population and shift following individuals up in list
                for i in range(ind2, len(self.pop)-1):
                    self.pop[i] = self.pop[i+1]
                self.pop.pop() # remove last element from list (because it was shifted up)

            elif self.dominates(ind2, ind1):  # ind2 dominates

                # remove ind1 from population and shift following individuals up in list
                for i in range(ind1, len(self.pop)-1):
                    self.pop[i] = self.pop[i+1]
                self.pop.pop() # remove last element from list (because it was shifted up)

        assert len(self.pop) == self.pop_size


    def dominates(self, ind1, ind2):
        # Returns true if ind1 dominates ind2, otherwise false
        if self.pop[ind1].age == self.pop[ind2].age and self.pop[ind1].fitness == self.pop[ind2].fitness:
            return self.pop[ind1].id > self.pop[ind2].id # if equal, return the newer individual

        elif self.pop[ind1].age <= self.pop[ind2].age and self.pop[ind1].fitness >= self.pop[ind2].fitness:
            return True
        else:
            return False
    

    # Parallelize the loop
    def evaluate(self, children):
        self.submit_batch(children)
        for child in children: 
            if np.abs(child.fitness_map['01']) == 0: child.fitness_map['01'] = 0.0001
            if np.abs(child.fitness_map['11']) == 0: child.fitness_map['11'] = 0.0001
            if np.abs(child.fitness_map['10']) == 0: child.fitness_map['10'] = 0.0001
            if np.abs(child.fitness_map['00']) == 0: child.fitness_map['00'] = 0.0001
            if self.gate == "AND":
                child.fitness = np.abs(child.fitness_map['11'])/((np.abs(child.fitness_map['01']) + np.abs(child.fitness_map['10']) + np.abs(child.fitness_map['00']))/3)
            if self.gate == "XOR":
                child.fitness = ((np.abs(child.fitness_map['01']) + np.abs(child.fitness_map['10']))/2)/((np.abs(child.fitness_map['00']) + np.abs(child.fitness_map['11']))/2)
            if self.gate == "OR":
                child.fitness = ((np.abs(child.fitness_map['01']) + np.abs(child.fitness_map['10']) + np.abs(child.fitness_map['11']))/3)/np.abs(child.fitness_map['00'])
            if self.gate == "NAND":
                child.fitness = ((np.abs(child.fitness_map['01']) + np.abs(child.fitness_map['10']) + np.abs(child.fitness_map['00']))/3)/np.abs(child.fitness_map['11'])
            child.fitness = np.log(child.fitness)


    def submit_batch(self, children):
        self.children = children 
        for input_case in ["01", "10", "11", '00']: #AND 
            with ProcessPoolExecutor() as executor:
                input_cases = [input_case]*len(self.children)
                results = executor.map(self.evaluate_material, range(len(self.children)), input_cases)

            for material_index, fitness in results:
                
                children[material_index].fitness_map[input_case] = fitness[self.target_particle]


    def evaluate_material(self, material_index, input_case, input_1=7, input_2=9):
            material = self.children[material_index]
            n_col = 4
            n_row = 3
            N = (n_col - 1) * n_row + int(np.floor(n_row / 2.0)) # total number of particles

            d0 = 1.
            y_top_disp = 0.01 ## compression amount
            Lx = d0 * n_col
            Ly = (n_row - 1) * np.sqrt(3) / 2 * d0 + d0

            D = np.ones(N, dtype = np.float64) * d0
            
            x = np.zeros(N, dtype = np.float64)
            y = np.zeros(N, dtype = np.float64)

            ind = -1
            for i_row in range(n_row):
                if i_row % 2 == 1:
                    n_col_now = n_col
                else:
                    n_col_now = n_col - 1
                for i_col in range(n_col_now):
                    ind += 1
                    if i_row % 2 == 1:
                        x[ind] = (i_col + 0.5) * d0
                    else:
                        x[ind] = (i_col + 1.) * d0
                    y[ind] = i_row * 0.5 * np.sqrt(3) * d0
            y = y + 0.5 * d0


            mass = np.ones(N, dtype = np.float64)
            k_list = np.ones(N, dtype = np.float64) * 1.
            for particle_index, particle in enumerate(material.particles): #fill in genome 
                D[particle_index] = d0 * (1. + particle.expansion) # inflate this specific particle
                k_list[particle_index] = particle.stiffness 
            
            
            if self.encoding == "size":
                if input_case == "00":
                    D[input_1] = 1.  #uninflated
                    D[input_2] = 1.  #uninflated
                    
                if input_case == "01": 
                    D[input_1] = 1.  #uninflated
                    D[input_2] = 1. + 0.04 #1 + max expansion 

                if input_case == "10": 
                    D[input_1] = 1. + 0.04 
                    D[input_2] = 1.  #uninflated

                if input_case == "11": 
                    D[input_1] = 1. + 0.04 
                    D[input_2] = 1. + 0.04 


            if self.encoding == "stiffness":
                if input_case == "00":
                    k_list[input_1] = 0.116279
                    k_list[input_2] = 0.116279
                    
                if input_case == "01": 
                    k_list[input_1] = 0.116279
                    k_list[input_2] = 1.0

                if input_case == "10": 
                    k_list[input_1] = 1.0
                    k_list[input_2] = 0.116279

                if input_case == "11": 
                    k_list[input_1] = 1.0
                    k_list[input_2] = 1.0

            Ly -= y_top_disp * d0
            Lx -= y_top_disp * d0

            FIRE_VL(N, x, y, D, Lx, Ly, k_list)


            Fx_w, Fy_w = ForceWall(N, x, y, D, Lx, Ly, k_list)
            # print(Fy_w[0])
            return material_index, Fy_w


    def plot_best(self, material, input_case, input_1=7, input_2=9):

        # material = self.children[material_index]
        n_col = 4
        n_row = 3
        N = (n_col - 1) * n_row + int(np.floor(n_row / 2.0)) # total number of particles

        d0 = 1.
        y_top_disp = 0.01 ## compression amount
        Lx = d0 * n_col
        Ly = (n_row - 1) * np.sqrt(3) / 2 * d0 + d0

        D = np.ones(N, dtype = np.float64) * d0


        x = np.zeros(N, dtype = np.float64)
        y = np.zeros(N, dtype = np.float64)

        ind = -1
        for i_row in range(n_row):
            if i_row % 2 == 1:
                n_col_now = n_col
            else:
                n_col_now = n_col - 1
            for i_col in range(n_col_now):
                ind += 1
                if i_row % 2 == 1:
                    x[ind] = (i_col + 0.5) * d0
                else:
                    x[ind] = (i_col + 1.) * d0
                y[ind] = i_row * 0.5 * np.sqrt(3) * d0
        y = y + 0.5 * d0


        mass = np.ones(N, dtype = np.float64)
        k_list = np.ones(N, dtype = np.float64) * 1.
        for particle_index, particle in enumerate(material.particles): #fill in genome 
            D[particle_index] = d0 * (1. + particle.expansion) # inflate this specific particle
            k_list[particle_index] = particle.stiffness 
       
        if self.encoding == "size":
            if input_case == "00":
                D[input_1] = 1.  #uninflated
                D[input_2] = 1.  #uninflated
                
            if input_case == "01": 
                D[input_1] = 1.  #uninflated
                D[input_2] = 1. + 0.04 #1 + max expansion 

            if input_case == "10": 
                D[input_1] = 1. + 0.04 
                D[input_2] = 1.  #uninflated

            if input_case == "11": 
                D[input_1] = 1. + 0.04 
                D[input_2] = 1. + 0.04 

                
        if self.encoding == "stiffness":
            if input_case == "00":
                k_list[input_1] = 0.116279
                k_list[input_2] = 0.116279
                
            if input_case == "01": 
                k_list[input_1] = 0.116279
                k_list[input_2] = 1.0

            if input_case == "10": 
                k_list[input_1] = 1.0
                k_list[input_2] = 0.116279

            if input_case == "11": 
                k_list[input_1] = 1.0
                k_list[input_2] = 1.0

            
        #add small pressure 
        Ly -= y_top_disp * d0
        Lx -= y_top_disp * d0

        FIRE_VL(N, x, y, D, Lx, Ly, k_list)

        ConfigPlot_DiffSize(N, x, y, D, Lx, Ly, k_list, 1, 0)
        Fx_w, Fy_w = ForceWall(N, x, y, D, Lx, Ly, k_list)

        plt.title(f'{Fy_w[self.target_particle]}')
        plt.savefig(f'best_configs_viz/seed{self.seed}_gen{self.curr_gen}_best_material_{material.id}_stiffness{self.change_stiffness}_size{self.change_size}_input{input_case}_encoding{self.encoding}_gate_{self.gate}')
        plt.close()


    def save_checkpoint(self):

        filename = 'checkpoints/{}_encoding/{}/run{}_gen{}_change_stiffness_{}_change_size_{}_gate{}_encoding{}.p'.format(self.encoding, self.gate, self.seed, self.curr_gen, self.change_stiffness, self.change_size, self.gate, self.encoding)
        print(filename)
        rng_state = random.getstate()
        np_rng_state = np.random.get_state()

        with open(filename, 'wb') as f:
            pickle.dump([self, rng_state, np_rng_state], f)

    
    def print_population(self, verbose=False):

        for i in range(len(self.pop)):

            self.pop[i].print(verbose=verbose)
        print()


    def print_best(self, verbose):
        best = self.return_best()
        print("BEST material:")
        best.print(verbose=verbose)
        self.plot_best(best, "11")
        self.plot_best(best, "00")
        self.plot_best(best, "10")
        self.plot_best(best, "01")



    def return_best(self):

        return sorted(self.pop, key=operator.attrgetter('fitness'), reverse=True)[0]
    