from model_representation.SeqEvoGenome import SeqEvoGenome
import math
import random 


class Crosser:  
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
        return SeqEvoGenome(first_half) 
        