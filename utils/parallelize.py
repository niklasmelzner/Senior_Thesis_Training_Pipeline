"""
Used to execute tasks in parallel
"""
import multiprocessing
from typing import Union


class TaskDistributor:
    """
    Handles synchronized distribution of tasks to available workers
    """

    def __init__(self, input: list, verbose: int):
        self.verbose = verbose
        self.input = input
        manager = multiprocessing.Manager()
        self.output = manager.list(range(len(input)))
        self.current_index = multiprocessing.Value("i", 0)
        self.distribution_lock = multiprocessing.Lock()
        self.done_count_lock = multiprocessing.Lock()
        self.done_count = multiprocessing.Value("i", 0)

    def get_next_input(self):
        """
        Called by a worker, returns the next input value or None if none is available
        """
        try:
            self.distribution_lock.acquire()
            index = self.current_index.value
            if len(self.input) <= index:
                return None, None
            self.current_index.value += 1
            return index, self.input[index]
        finally:
            self.distribution_lock.release()

    def add_output(self, index: int, value):
        """
        Called by a worker to report the result of an execution
        """
        self.output[index] = value
        if self.verbose > 0:
            try:
                self.done_count_lock.acquire()
                self.done_count.value += 1
                print(str(self.done_count.value) + "/" + str(len(self.input)) + " done")
            finally:
                self.done_count_lock.release()


def compute_in_parallel_worker(task: callable, task_distributor: TaskDistributor, args: Union[dict, list, tuple]):
    """
    Processes inputs from the task distributor as long as available
    """
    while True:
        index, input = task_distributor.get_next_input()
        if input is None:
            # all inputs distributed, terminate
            return
        if type(args) == dict:
            task_distributor.add_output(index, task(input, **args))
        else:
            task_distributor.add_output(index, task(input, *args))


def compute_in_parallel(input: list, task: callable, n_workers: int = -1, verbose: int = 1,
                        args: Union[dict, list, tuple] = None):
    """
    Executes the given task for each input value and returns the results in the same order as the input values
    """
    if args is None:
        args = {}
    if n_workers < 0:
        n_workers = multiprocessing.cpu_count()

    # for distributing tasks and handling return values
    task_distributor = TaskDistributor(input, verbose=verbose)

    # start processes
    processes = []
    for i in range(n_workers):
        process = multiprocessing.Process(target=compute_in_parallel_worker, args=(task, task_distributor, args))
        process.start()
        processes.append(process)

    # wait for processes to finish
    for process in processes:
        if process.is_alive():
            process.join()

    return task_distributor.output
