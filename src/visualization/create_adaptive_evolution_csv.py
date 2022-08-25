from importlib.resources import path
from pathlib import Path
import re
import csv

def find_all(input_str, search_str):
    return [m.start() for m in re.finditer(search_str, input_str)]

def start_end_cut(input_str, start_str, end_str, start_add=0, end_add=0, start_n_of_occurence=1, end_n_of_occurence=1):
    start_idx = None
    if start_n_of_occurence == 1:
        start_idx = input_str.index(start_str)
    else:
        start_idx = find_all(input_str=input_str, search_str=start_str)[start_n_of_occurence-1]
    start_idx += start_add
    subtext = input_str[start_idx:]

    end_idx = None
    if end_n_of_occurence == 1:
        end_idx = subtext.index(end_str)
    else:
        end_idx = find_all(input_str=subtext, search_str=end_str)[end_n_of_occurence-1]
    end_idx += end_add
    return subtext[:end_idx]


def get_generation_str(input_str, n_generation):
    return start_end_cut(
        input_str=input_str,
        start_str=f"======================= Generation {n_generation}",
        end_str=f"======================= Generation {n_generation+1}",
        end_add=-50
    )

def get_best_individual_str(generation_str):
    return start_end_cut(
        input_str=generation_str,
        start_str="*** Ranking:",
        start_add=14,
        end_str="SeqEvoGenome",
        end_n_of_occurence=2
    )

def get_best_individual_fitness(best_individual_str):
    return float(start_end_cut(
        input_str=best_individual_str,
        start_str="fitness:",
        start_add=9,
        end_str="created_from:",
        end_add=-2
    ))

def get_current_fitness_threshold(generation_str):
    return float(start_end_cut(
        input_str=generation_str,
        start_str="Fitness threshold:",
        start_add=19,
        end_str="->",
    ))

def get_optimization_stage(generation_str):
    return start_end_cut(
        input_str=generation_str,
        start_str="Optimization stage:",
        start_add=20,
        end_str="->",
    )
    

def get_data_for_gen(input_str, n_generation):
    generation_str = get_generation_str(input_str=input_str, n_generation=n_generation)
    best_individual_str = get_best_individual_str(generation_str=generation_str)

    return {
        "best_individual_fitness": get_best_individual_fitness(best_individual_str=best_individual_str), 
        "current_fitness_threshold": get_current_fitness_threshold(generation_str=generation_str),
        "optimization_stage": get_optimization_stage(generation_str=generation_str)  
    }

def write_row(path_to_file, data_row):
    with open(path_to_file, 'a+', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(data_row)

def create_adaptive_csv_from_logs(path_to_logs, path_to_csv, max_generation):

    log_str = Path(path_to_logs).read_text()

    # Header
    header_row = ["n_generation", "best_individual_fitness", "current_fitness_threshold", "optimization_stage"]
    write_row(path_to_file=path_to_csv, data_row=header_row)

    # Data
    for n_generation in range(1, max_generation):
        data = get_data_for_gen(input_str=log_str, n_generation=n_generation)
        data_row = [
            n_generation,
            data["best_individual_fitness"], 
            data["current_fitness_threshold"],
            data["optimization_stage"]
        ]
        write_row(path_to_file=path_to_csv, data_row=data_row)

if __name__ == "__main__":
    
    create_adaptive_csv_from_logs(
        path_to_logs='tmp/eva2.txt',
        path_to_csv="tmp/adaptive_evolution.csv",
        max_generation=142
    )

