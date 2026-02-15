import time
from datetime import timedelta
from typing import Callable


def timer(print_message: str = "") -> Callable:
    """A decorator to log the time taken by a function.

    Args:
        print_message: An optional custom message to be logged along with the elapsed time.

    Returns:
        A decorator that takes a callable and returns a wrapped callable that logs its execution time.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: any, **kwargs: any) -> any:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = timedelta(seconds=end_time - start_time)
            formatted_time = str(elapsed_time).split(".")[0]
            if print_message:
                print(f"{print_message} took: {formatted_time}")
            else:
                print(f"Total time taken: {formatted_time}")
            return result

        return wrapper

    return decorator
