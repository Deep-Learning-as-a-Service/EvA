from optimizer.SeqEvo.EvoTechnique import EvoTechnique

class EvoTechniqueConfig():

    @property
    def techniques(self) -> 'list[EvoTechnique]':
        assert self._techniques is not None, "subclass of EvoTechniqueConfig needs to define self.techniques"
        return self._techniques
    
    @property
    def mid_optimization_start_gen(self) -> int:
        assert self._mid_optimization_start_gen is not None, "subclass of EvoTechniqueConfig needs to define _mid_optimization_start_gen"
        return self._mid_optimization_start_gen
    
    @property
    def micro_optimization_start_gen(self) -> int:
        assert self._micro_optimization_start_gen is not None, "subclass of EvoTechniqueConfig needs to define _micro_optimization_start_gen"
        return self._micro_optimization_start_gen

class DefaultEvoTechniqueConfig(EvoTechniqueConfig):
    def __init__(self):
        self._mid_optimization_start_gen = 30
        self._micro_optimization_start_gen = 100
        self._techniques = [
            EvoTechnique(
                name="mutate_low",
                optimization_stage="micro"
            ),
            EvoTechnique(
                name="mutate_mid",
                optimization_stage="mid"
            ), 
            EvoTechnique(
                name="mutate_high",
                optimization_stage="mid"
            ),
            EvoTechnique(
                name="mutate_all",
                optimization_stage="macro"
            ),
            EvoTechnique(
                name="crossover",
                optimization_stage="mid"
            ),
            EvoTechnique(
                name="finetune_best_individual",
                optimization_stage="micro"
            ),
            EvoTechnique(
                name="random_default",
                optimization_stage = "none"
            )
        ]