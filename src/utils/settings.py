import os


def init(_layer_pool, _data_dimension_dict):
    """
    Refactoring idea:
    - pass the mapping, that we can easily switch between datasets and labels
    - mapping.py file (in utils) should include activity and subject mappings for the datasets
    - the experiments loads the required ones and passes them in the init (settings.init(mappings)O
    """

    global input_distribution_mean
    input_distribution_mean = None
    
    global input_distribution_variance
    input_distribution_variance = None
    
    global data_dimension_dict
    data_dimension_dict = _data_dimension_dict

    global layer_pool
    layer_pool = _layer_pool

    global opportunity_dataset_path
    opportunity_dataset_path = "data/opportunity-dataset"

    global opportunity_dataset_csv_path
    opportunity_dataset_csv_path = "data/opportunity-dataset-csv"
    
    global pamap2_dataset_path
    pamap2_dataset_path = "data/pamap2-dataset/Protocol"
    
    global pamap2_initial_num_to_activity_idx
    pamap2_initial_num_to_activity_idx = {
        1:0, 
        2:1, 
        3:2, 
        4:3, 
        16:4, 
        17:5
        
    }
    
    global activity_initial_num_to_activity_str
    activity_initial_num_to_activity_str = {
        0: "null",
        101: "relaxing",
        102: "coffee time",
        103: "early morning",
        104: "cleanup",
        105: "sandwich time",
    }

    global activity_initial_num_to_activity_idx
    activity_initial_num_to_activity_idx = {
        0: 0,
        101: 1,
        102: 2,
        103: 3,
        104: 4,
        105: 5,
    }

    global saved_experiments_path
    saved_experiments_path = 'src/saved_experiments'

