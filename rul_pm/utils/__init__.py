from typing import Iterable, Optional

from tqdm.auto import tqdm

show_progressbar = True


def progress_bar(iterable: Iterable, n: Optional[int] = None):
    if show_progressbar:
        return tqdm(iterable, total=n)
    else:
        return iterable
