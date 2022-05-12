import random

def get_mutation_probability(rand_dict) -> str:
        assert sum(rand_dict.values()) == 1.0, "sum of probabilities should be 1"
        rand_number = random.random()
        mutation: str = None
        endProbability = 0.0
        for mutation_type, mutation_type_probability in rand_dict.items():
            endProbability += mutation_type_probability
            if rand_number <= endProbability:
                mutation = mutation_type
                break
        return mutation