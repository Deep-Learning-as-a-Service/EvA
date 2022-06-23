from model_representation.EvoParam.DEvoParam import DEvoParam
from model_representation.EvoParam.EvoParam import EvoParam
from model_representation.ParametrizedLayer.ParametrizedLayer import ParametrizedLayer
from model_representation.ParametrizedLayer.PConv1DLayer import PConv1DLayer
from model_representation.ParametrizedLayer.PConv2DLayer import PConv2DLayer
from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer
from model_representation.ParametrizedLayer.PLstmLayer import PLstmLayer

from optimizer.SeqEvo.HistorySeqEvoGenome import HistorySeqEvoGenome
import time
import csv
import os
import pandas as pd
import re
import sys
from ast import literal_eval as make_tuple
from optimizer.SeqEvo.SeqEvoGenome import SeqEvoGenome

class SeqEvoHistory:
    def __init__(self, path_to_file):
        self.path_to_file = path_to_file
        self.header = ['created_unix_timestamp', 'layer_list', 'fitness', 'created_from', 'n_generations', 'parent_fitness']
        self.file_exists = os.path.isfile(self.path_to_file)
    
    def _write_row(self, data_row) -> None:
        assert len(data_row) == len(self.header)
        
        if not self.file_exists:
            with open(self.path_to_file, 'a+', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(self.header)
            self.file_exists = True

        with open(self.path_to_file, 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data_row)
    
    def write(self, seqevo_genome, n_generations) -> None:
        created_unix_timestamp = time.time()
        layer_list = seqevo_genome.layer_list_str()
        fitness = seqevo_genome.fitness
        created_from = seqevo_genome.created_from
        parent_fitness = seqevo_genome.parent_fitness

        data_row = [created_unix_timestamp, layer_list, fitness, created_from, n_generations, parent_fitness]
        self._write_row(data_row)


    def read(self) -> 'list[HistorySeqEvoGenome]':
        data = pd.read_csv(self.path_to_file, sep=',')
        history_seqevo_genomes = []
        for idx in range(len(data)):
            dataline = data.iloc[idx]
            layers = self._extract_layers(dataline['layer_list'])
            history_seqevo_genomes.append(HistorySeqEvoGenome(
                seqevo_genome=SeqEvoGenome(layers=layers), 
                fitness=dataline['fitness'], 
                created_from=dataline['created_from'], 
                n_generations=dataline['n_generations'],
                created_unix_timestamp = dataline['created_unix_timestamp'],
                src_file = self.path_to_file,
                parent_fitness = dataline['parent_fitness']
                ))
            
        return history_seqevo_genomes

    def _extract_layers(self, layer_list_string: str) -> 'list[ParametrizedLayer]':
        layers = []
        
        # remove whitespaces from tuples so i can extract the different params via whitespaces later
        reg = r"\d,\s\d"
        matches = re.finditer(reg, layer_list_string)
        for idx, match in enumerate(matches):
            whitespace_idx = match.span()[1] - 1
            layer_list_string = layer_list_string[:whitespace_idx - 1 - idx] + layer_list_string[whitespace_idx - idx:]
            
        # split layer_list by regex thats looks for closed parenthesis followed by a whitespace and P.....Layer
        reg = r'\)\s(P.*?Layer)'
        split_indices = [match.span()[0]+2 for match in re.finditer(reg, layer_list_string)]
        split_indices.insert(0,0)
        layer_list = [layer_list_string[i:j] for i,j in zip(split_indices, split_indices[1:]+[None])]
        
        for idx, layer in enumerate(layer_list):
            
            # delete whitespace suffix from layerstrings
            if idx != len(layer_list) - 1:
                layer = layer[:-1]
            
            # get class name and class object by taking the start string until first opening parenthesis
            reg = r"^[^\(]*"
            class_name = re.match(reg, layer)[0]
            class_obj = getattr(sys.modules[__name__], class_name)
            
            # remove class name, open parenthesis and closing parenthesis from layer string
            layer = layer.replace(class_name + '(', '')
            layer = layer[:-1]
            
            # parse params and param values from layer string (IMPORTANT: only ints and tuples are supported yet)
            attr_value_dict = {}
            for attribute_str in layer.split(" "):
                attribute_tuple = attribute_str.split("=")
                value = int(attribute_tuple[1]) if attribute_tuple[1].isdigit() else make_tuple(attribute_tuple[1])
                attr_value_dict[attribute_tuple[0]] = value
                
            params = []
            
            # first only instantiate EvoParams and NOT DEvoParams
            for param_class in class_obj._param_classes:
                if not issubclass(param_class, DEvoParam):
                    value = attr_value_dict[param_class._key]
                    params.append(param_class.create(value))
                    
            # then instantiate DEvoParams by looking up the corresponding EvoParam that it is dependent on (IMPORTANT: only ints and tuples are supported yet)
            for param_class in class_obj._param_classes:
                if issubclass(param_class, DEvoParam):
                    value = attr_value_dict[param_class._key]
                    dependent_class = param_class._dependent_class
                    dependent_obj = list(filter(lambda param: isinstance(param, dependent_class), params))[0]
                    categorical_tuple = self._extract_dependent_value(value, dependent_obj.value)
                    params.append(param_class.create(dependent_value=categorical_tuple, dependent_on_param=dependent_obj))
            layers.append(class_obj.create_from_params(params))
            
        return layers
    
    def _extract_dependent_value(self, value, dependent_value):
        def get_percentage_categorical(value, dependent_value):        
            if value == 1:
                return "step1"
            # very specific to 25%, 50%, 75%, 100% categoricals
            value = 4 * value
            dependent_value = dependent_value
            return str(round(value / dependent_value) * 25) + "%"
            
        if isinstance(dependent_value, tuple):
            return (get_percentage_categorical(value[0], dependent_value[0]), get_percentage_categorical(value[1], dependent_value[1]))
        if isinstance(dependent_value, int):
            return get_percentage_categorical(value, dependent_value)