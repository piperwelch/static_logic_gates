from afpo import AFPO 
import sys

SEED = int(sys.argv[1])
GATE = str(sys.argv[2])
encoding = str(sys.argv[3])
change_stiffness = bool(int(sys.argv[4]))
change_size = bool(int(sys.argv[5]))
# print(change_stiffness)
# print(change_size)
# print(sys.argv[4])
# print(sys.argv[5])

# quit()
GENS = 1
POPSIZE = 1000 # popsize needs to be large for AFPO
CHECKPOINT_EVERY = 10 # number of gens to save out state of afpo

target_particle = 1

afpo = AFPO(random_seed=SEED, gens=GENS, pop_size=POPSIZE, checkpoint_every=CHECKPOINT_EVERY, change_stiffness=change_stiffness, change_size=change_size, target_particle=target_particle, gate=GATE, encoding=encoding)

afpo.run()
