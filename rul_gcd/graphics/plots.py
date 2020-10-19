import seaborn as sns
import matplotlib.pyplot as plt
from rul_gcd.iterators import LifeDatasetIterator


def plot_lives(ds):
    fig, ax = plt.subplots()
    it = LifeDatasetIterator(ds)
    for _, y in it:
        ax.plot(y)
    return fig, ax