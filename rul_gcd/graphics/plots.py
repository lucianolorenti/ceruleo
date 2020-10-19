import seaborn as sns
import matplotlib.pyplot as plt
from rul_gcd.iterators.iterators import LifeDatasetIterator
from rul_gcd.dataset.lives_dataset import AbstractLivesDataset


def plot_lives(ds : AbstractLivesDataset):
    """
    Plot each life
    """
    fig, ax = plt.subplots()
    it = LifeDatasetIterator(ds)
    for _, y in it:
        ax.plot(y)
    return fig, ax