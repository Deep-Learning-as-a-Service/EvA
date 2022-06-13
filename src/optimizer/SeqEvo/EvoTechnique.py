from dataclasses import dataclass

technique_names = ["mutate_low", "mutate_mid", "mutate_high", "mutate_all", "crossover", "finetune_best_individual", "random_default"]
optimization_stages = ["micro", "mid", "macro", "none"]

@dataclass
class EvoTechnique():
    def __init__(self, name, optimization_stage):
        """
        Tracker for adaptive evolution
        - will start with applying macro techniques preferred, end with micro techniques

        TODO: 
        - if evolution gets stuck at some point, this information can be used to change techniques
        - add pool technique
        - idea: do a run with all techniques applied equally to get an unbiased evaluation at what point what technique is good

        name: "mutation mid" | "uniform crossover" | ...
        optimization_stage: "micro" | "mid" | "macro" (optimization)
        n_improvements: (how often has this technique came up with a new individual)
        improvements: list[Tuple(n_gen, improvement_amount)], e.g.: [(1, 0.2), (5, 0.02), (5, 0.001), (7, 0.3)] 
        fitnesses: e.g. [[0.3, 0.1], [0.5, 0.2], [0.5], [0.2, 0.4, 0.5]] (list for each generation, fitness_value for each individual created in this generation)
        """
        assert name in technique_names, f"technique_name {name} is not a listed technique name"
        self.name = name

        assert optimization_stage in optimization_stages, f"optimization_stage {optimization_stage} is not a listed optimization_stage"
        self.optimization_stage = optimization_stage

        self.improvements = []
        self.fitnesses = []