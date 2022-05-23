import signal
import sys
import time
import traceback

from utils.telegram import send_telegram


def error_after_seconds(time):
    def decorator(function):
        def wrapper(*args, **kwargs):
            def raise_timeout(signum, frame):
                raise TimeoutError

            signal.signal(signal.SIGALRM, raise_timeout)
            signal.alarm(time)
            try:
                return function(*args, **kwargs)
            finally:
                # Unregister the signal so it won't be triggered
                # if the timeout is not reached.
                signal.signal(signal.SIGALRM, signal.SIG_IGN)

        return wrapper

    return decorator

def on_error_restart(log_func):
    def on_error_restart_decorator(function):
        def wrapper_function(*args, **kwargs):
            while True:
                try:
                    return function(*args, **kwargs)
                except Exception as err:
                    wait_seconds = 310
                    log_func("**** Programm Execution failed ****\n- waiting " + str(wait_seconds) + " seconds before restarting")
                    time.sleep(wait_seconds)
                    log_func("**** Restarting Programm ****")
                    
        return wrapper_function
    return on_error_restart_decorator

def on_error_send_traceback(log_func):
    def on_error_send_traceback_decorator(function):
        def wrapper_function(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as err:
                # traceback.print_tb(err.__traceback__)
                etype, value, tb = sys.exc_info()
                max_stack_number = 300
                traceback_string = ''.join(traceback.format_exception(etype, value, tb, max_stack_number))
                log_func('Exception in ' + function.__name__ + '\n' + traceback_string)
                raise err

        return wrapper_function
    return on_error_send_traceback_decorator

def log_job_start_done(log_func):
    def job_start_done_decorator(function):
        def wrapper_function(*args, **kwargs):
            log_func('-------- Job started: ' + function.__name__)
            return_value = function(*args, **kwargs)
            log_func('-------- Job done: ' + function.__name__)
            return return_value
        return wrapper_function
    return job_start_done_decorator