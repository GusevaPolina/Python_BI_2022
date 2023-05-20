##############################################################################
# this homework explores the functionality of parallel programming in Python #
##############################################################################
from typing import Optional, List, Tuple, Any, Callable, Dict, Iterable, Union

# subtask 1: Custom Random Forest with parallel calculations
import numpy as np
import warnings
import random

from sklearn.tree import (DecisionTreeRegressor,
                          DecisionTreeClassifier)
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification

from joblib import Parallel, delayed
from multiprocessing import Pool

# establish seed and silence warnings
warnings.filterwarnings("ignore")

SEED: int = 111
random.seed(SEED)
np.random.seed(SEED)


class RandomForestClassifierCustom(BaseEstimator):
    """
    A custom implementation of RandomForestClassifier using scikit-learn's BaseEstimator
    """

    def __init__(self,
                 n_estimators: int = 10,
                 max_depth: Optional[int] = None,
                 max_features: Optional[int] = None,
                 random_state: int = SEED) -> None:
        """
        Initialization method for RandomForestClassifierCustom

        :param n_estimators: int, optional (default=10)
            The number of decision trees in the forest.
        :param max_depth: int, optional (default=None)
            The maximum depth of each decision tree. If None, then nodes are expanded until all the leaves are pure.
        :param max_features: int, optional (default=None)
            The number of features to consider when looking for the best split. If None, then all features will be used.
        :param random_state: int, optional (default=SEED)
            Seed used by the random number generator.

        :return: None
        """
        self.classes_ = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        self.feat_ids_by_tree = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_jobs: int = 1) -> "RandomForestClassifierCustom":
        """
        Build a forest of decision trees from the training set (X, y).

        :param X: array-like of shape (n_samples, n_features)
            The training input samples.
        :param y: array-like of shape (n_samples,)
            The target values (class labels) as integers or strings.
        :param n_jobs: int, optional (default=1)
            The number of jobs to run in parallel for both `feat_ids_by_tree` and `trees` formation.

        :return: RandomForestClassifierCustom
            Fitted estimator instance
        """
        self.classes_ = sorted(np.unique(y))

        def tree_step_maker(i: int) -> np.ndarray:
            np.random.seed(self.random_state + i)
            feat_ids_by_tree_step = np.random.choice(np.arange(X.shape[1]),
                                                     size=self.max_features,
                                                     replace=False)
            return feat_ids_by_tree_step

        self.feat_ids_by_tree = Parallel(n_jobs=n_jobs)(delayed(tree_step_maker)(i) for i in range(self.n_estimators))

        def trees_maker(i: int) -> DecisionTreeClassifier:
            np.random.seed(self.random_state + i)
            pseudo_slice = np.random.choice(np.arange(len(X)),
                                            size=len(X), replace=True)

            clf = DecisionTreeClassifier(max_depth=self.max_depth,
                                         max_features=self.max_features,
                                         random_state=self.random_state + i)

            return clf.fit(X[pseudo_slice][:, self.feat_ids_by_tree[i]], y[pseudo_slice])

        self.trees = Parallel(n_jobs=n_jobs)(delayed(trees_maker)(i) for i in range(self.n_estimators))

        return self

    def _pred_maker(self, feats: np.ndarray,
                    tree: DecisionTreeClassifier,
                    X: np.ndarray) -> np.ndarray:
        return tree.predict_proba(X[:, feats])

    def predict_proba(self, X: np.ndarray, n_jobs: int = 2) -> np.ndarray:
        """
        Predict class probabilities of input data using a single decision tree.

        :param n_jobs: the number of parallel jobs to run
        :param X: array-like of shape (n_samples, n_features)
            The input data.

        :return: np.ndarray
            Array of predicted probabilities.
        """

        with Pool(n_jobs) as pool:
            y_pred_storage = pool.starmap(self._pred_maker,
                                          [(feats, tree, X) for feats, tree in zip(self.feat_ids_by_tree, self.trees)])

        return np.average(y_pred_storage, axis=0)

    def predict(self, X: np.ndarray, n_jobs: int = 1) -> np.ndarray:
        """
        Predict class probabilities of input data using a single decision tree.

        :param n_jobs: the number of parallel jobs to run
        :param X: array-like of shape (n_samples, n_features)
            The input data.

        :return: np.ndarray
            Array of the most likely outcome.
        """
        probas = self.predict_proba(X, n_jobs=n_jobs)
        predictions = np.argmax(probas, axis=1)

        return predictions


# subtask 2: the memory_limit decorator that caps the memory usage
import os
import psutil
import time
import warnings
import threading


def get_memory_usage() -> int:
    """
    Returns the current memory usage of the process in bytes.

    Returns:
        int: Current memory usage of the process in bytes.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def bytes_to_human_readable(n_bytes: int) -> str:
    """
    Converts a byte count to a human-readable string.

    Args:
        n_bytes (int): Number of bytes to convert.

    Returns:
        str: A string representing the input bytes count in human-readable format.
    """
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for idx, s in enumerate(symbols):
        prefix[s] = 1 << (idx + 1) * 10
    for s in reversed(symbols):
        if n_bytes >= prefix[s]:
            value = float(n_bytes) / prefix[s]
            return f"{value:.2f}{s}"
    return f"{n_bytes}B"


def human_to_bytes_readable(human_size: str) -> float:
    """
    Converts a human-readable byte count to an integer.

    Args:
        human_size (str): A string representing the byte count in human-readable format.

    Returns:
        float: The equivalent byte count as a floating point number.
    """
    symbols = ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for idx, s in enumerate(symbols):
        prefix[s] = 1 << (idx) * 10

    return float(human_size[:-1]) * prefix[human_size[-1]]


def memory_limit(hard_limit: Optional[str] = None,
                 soft_limit: Optional[str] = None,
                 poll_interval: Optional[float] = None) -> Callable:
    """
    A decorator that limits the memory usage of a function by crashing it if it exceeds the specified limits.

    Args:
        hard_limit: A string representing the maximum memory usage allowed by the function in human-readable format.
            If None, the hard limit will be twice the current memory usage.
        soft_limit: A string representing the memory usage threshold for a warning message in human-readable format.
            If None, no warning will be issued.
        poll_interval: The interval in seconds between memory usage checks.

    Returns:
        The decorated function.

    Raises:
        MemoryError: If the memory usage exceeds the hard limit.
    """

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            def check_memory_usage(hard_limit: Optional[str] = hard_limit,
                                   soft_limit: Optional[str] = soft_limit,
                                   poll_interval: Optional[float] = poll_interval) -> None:
                """
                Check the memory usage of the current process periodically and raise an error when the hard memory limit is exceeded.

                :param hard_limit: A string representing the hard memory limit, e.g. "1G".
                                  If not provided, it defaults to 2 times the current memory usage.
                :param soft_limit: A string representing the soft memory limit, e.g. "500M".
                                  If provided, a warning is issued when the memory usage exceeds this limit.
                :param poll_interval: The time interval between memory usage checks, in seconds.
                                      If not provided, it defaults to 1 second.
                """
                print('check starts \n')
                flag = True

                if hard_limit:
                    hard_lim_set = human_to_bytes_readable(hard_limit)
                else:
                    hard_lim_set = 2 * get_memory_usage()

                while get_memory_usage() < hard_lim_set:
                    time.sleep(poll_interval)

                    if flag and soft_limit:
                        if get_memory_usage() > human_to_bytes_readable(soft_limit):
                            now = bytes_to_human_readable(get_memory_usage())
                            warnings.warn(
                                f'Warning: the soft memory limit of {soft_limit} is exceeded: the current memory usage is {now}')
                            flag = False

                    if not hard_limit:
                        hard_lim_set = 2 * get_memory_usage()

                now = bytes_to_human_readable(get_memory_usage())
                print(f'\nThe hard memory limit of {hard_limit} is exceeded: the current memory usage is {now}')
                print('The session will be crashed in 0.3 seconds....')

                time.sleep(0.3)
                # raise MemoryError('Memory usage exceeded limit')
                os._exit(1)

            thread = threading.Thread(target=check_memory_usage)
            thread.start()

            result = func(*args, **kwargs)

            thread.join()
            print('the decorator is finished')

            return result

        return wrapper

    return decorator


@memory_limit(soft_limit="512M", hard_limit="1.2G", poll_interval=0.1)
def memory_increment():
    """
    Функция для тестирования

    В течение нескольких секунд достигает использования памяти 1.89G
    Потребление памяти и скорость накопления можно варьировать, изменяя код
    """
    print('hehe i have started\n')
    lst = []
    for i in range(40000000):
        if i % 400000 == 0:
            time.sleep(0.1)
        lst.append(i)

    print('hehe u sucks')
    return lst


# subtask 3: a universal function that parallels everything!
import multiprocessing
import time


def parallel_map(target_func: Callable[..., Any],
                 args_container: List[Iterable] = None,
                 kwargs_container: List[Dict[str, Any]] = None,
                 n_jobs: int = None) -> Iterable[Tuple[Iterable, Dict[str, Any], Any]]:
    """
    Applies a function to a list of arguments in parallel using multiple processes.

    Args:
        target_func (Callable): The function to be applied in parallel.
        args_container (List[Iterable], optional): A list of argument iterables to be passed to the function.
            Defaults to None.
        kwargs_container (List[Dict[str, Any]], optional): A list of keyword argument dictionaries to be passed to
            the function. Defaults to None.
        n_jobs (int, optional): The number of parallel processes to use. Defaults to None, which means that
            the number of processes used will be equal to the number of available CPU cores.

    Returns:
        An iterable of tuples containing the input arguments, keyword arguments, and results of applying the function.

    Raises:
        ValueError: If the lengths of `args_container` and `kwargs_container` do not match.

    """

    n_jobs = (multiprocessing.cpu_count(), n_jobs)[int(bool(n_jobs))]

    if args_container or kwargs_container:
        if args_container is None:
            args_container = [()] * len(kwargs_container)
        if kwargs_container is None:
            kwargs_container = [{}] * len(args_container)

        if len(args_container) != len(kwargs_container):
            raise ValueError("Lengths of args_container and kwargs_container do not match")

    print('lengths checked')

    task_queue = multiprocessing.Queue()

    print('queue made')

    if args_container or kwargs_container:
        for args, kwargs in zip(args_container, kwargs_container):
            task_queue.put((target_func, args, kwargs))
    else:
        task_queue.put(target_func)

    print('args, kwargs sorted')

    def run_task(task: Tuple[Callable[..., Any], Iterable,
                             Dict[str, Any]]) -> Tuple[Iterable, Dict[str, Any], Any]:
        target_func, args, kwargs = task
        return args, kwargs, target_func(*args, **kwargs)

    with multiprocessing.Pool(n_jobs) as pool:
        results = pool.imap_unordered(run_task, task_queue)

    print('multiprocessing done')

    return results


# Это только один пример тестовой функции, ваша parallel_map должна уметь эффективно работать с ЛЮБЫМИ функциями
# Поэтому обязательно протестируйте код на чём-нибудбь ещё
def test_func(x: Union[int, float, complex] = 1,
              s: Union[int, float, complex] = 2,
              a: Union[int, float, complex] = 1,
              b: Union[int, float, complex] = 1,
              c: Union[int, float, complex] = 1):
    time.sleep(s)
    return a*x**2 + b*x + c
