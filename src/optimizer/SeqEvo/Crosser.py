from optimizer.SeqEvo.SeqEvoGenome import SeqEvoGenome
import math
import random 


class Crosser():  
    @staticmethod
    def middlepoint_crossover(ma: SeqEvoGenome, pa: SeqEvoGenome):
        
        # get random int from [0, 1] and find halves of both parents
        random_int = round(random.random())
        middle_ma = math.floor(len(ma.layers)/2)
        middle_pa = math.floor(len(pa.layers)/2)
        
        # if random_int == 0: ma first half, pa last half
        # if random_int == 1: pa first half, ma second half
        first_half = ma.layers[:middle_ma] if random_int == 0 else pa.layers[:middle_pa]
        second_half = ma.layers[middle_ma:] if random_int == 1 else pa.layers[middle_pa:]
        first_half.extend(second_half)
        return SeqEvoGenome(first_half, created_from="middlepoint_crossover", parents=[ma, pa]) 
    
    @staticmethod
    def uniform_crossover(ma: SeqEvoGenome, pa: SeqEvoGenome):
        child_layers = []
        lowest_length = min(len(ma.layers), len(pa.layers))
        
        # randomize which layer the child gets from which parent layerwise
        for i in range(lowest_length):
            child_layers.append(random.choice([ma.layers[i], pa.layers[i]]))
            
        # excessive layers will get added randomly or not (50/50)
        longer_parent = ma if len(ma.layers) > len(pa.layers) else pa
        for i in range(lowest_length, len(longer_parent.layers)):
            if round(random.random()) == 0:
                child_layers.append(longer_parent.layers[i])
        return SeqEvoGenome(child_layers, created_from="uniform_crossover", parents=[ma, pa])
            
        