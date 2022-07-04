"""
File that gets executed!
Only import from experiments and tests
"""
from utils.server_manager import on_error_send_traceback, log_job_start_done, on_error_restart
from utils.telegram import send_telegram
from tensorflow.python.client import device_lib
from utils.logger import logger

prio_logger = lambda *args, **kwargs: logger(*args, prio=True, path="logs/general_logs", **kwargs)

# @on_error_restart(log_func=prio_logger)
# @on_error_send_traceback(log_func=prio_logger)
@log_job_start_done(log_func=prio_logger)
def main():
    # import tests.test_seqevo
    # import experiments.recreate_acc
    # import experiments.fitness_func_evaluation
    # import experiments.find_best_fitness
    # import experiments.analyse_history
    # import experiments.analyse_history
    # import experiments.competing_architectures_big_split
    # import tests.test_get_data
    import experiments.activationnormalisation_artemis

if __name__ == "__main__":
    prio_logger(device_lib.list_local_devices())
    main()
    