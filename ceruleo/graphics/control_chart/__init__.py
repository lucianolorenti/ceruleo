from abc import abstractmethod
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


class ControlChart:
    
    def violating_points(self, data):
        pass


    @abstractmethod
    def plot(self, data: np.ndarray, ax: plt.Axes):
        pass


class CChart(ControlChart):
    @classmethod
    def build(cls, data):
        cbar = np.mean(data)

        lcl = cbar - 3 * np.sqrt(cbar)
        ucl = cbar + 3 * np.sqrt(cbar)
        return cbar, lcl, ucl


def compute_ewma(data: np.ndarray, target: float, weight: float) -> np.ndarray:
    """Compute the Exponentially Weighted Moving Average (EWMA) of a dataset.

    Args:
        data (np.ndarray): The data to be analyzed.
        target (float): The target value.
        weight (float): The weight to be applied.

    Returns:
        np.ndarray: The EWMA values.
    """
    ewma_values = [target]
    for i in range(1, len(data)):
        ewma_values.append(weight * data[i] + (1 - weight) * ewma_values[-1])
    return np.array(ewma_values)


class EWMAChart(ControlChart):
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    target: float
    ewma: np.ndarray

    def __init__(
        self,
        *,
        ewma: np.ndarray,
        target: float,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
    ):
        self.ewma = ewma
        self.target = target
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def plot(self, data: np.ndarray, ax: plt.Axes) -> plt.Axes:
        ax.plot(data, label="data")
        ax.plot(self.ewma, label="ewma")
        ax.axhline(self.target, color="r", linestyle="--", label="target")
        ax.plot(self.lower_bound, color="g", linestyle="--", label="lcl")
        ax.plot(self.upper_bound, color="g", linestyle="--", label="ucl")
        # Colored region 
        ax.fill_between(
            np.arange(len(data)),
            self.lower_bound,
            self.upper_bound,
            color="gray",
            alpha=0.5,
        )
        ax.legend(
            loc="upper left",
            fontsize="small",
            title_fontsize="small",
            title="Control Limits",
        )
        return ax

    @classmethod
    def build(
        cls,
        data: np.ndarray,
        *,
        target: Optional[float] = None,
        weight: float = 0.2,
        k: float = 2.66,
    ):
        """Build an Exponentially Weighted Moving Average (EWMA) control chart.

        Args:
            data (np.ndarray): The data to be analyzed.
            target (Optional[float], optional): . Defaults to None.
            weight (float, optional): _description_. Defaults to 0.2.
            k (float, optional): _description_. Defaults to 2.66.

        Returns:
            _type_: _description_
        """
        if target is None:
            target = np.mean(data)

        std = np.std(data)
        n = len(data)
        ewma = compute_ewma(data, target, weight)

        upper_bound = ewma + k * std * np.sqrt(
            (1 - (1 - weight) ** (2 * np.arange(1, n + 1))) / (1 - (1 - weight) ** (2))
        )
        lower_bound = ewma - k * std * np.sqrt(
            (1 - (1 - weight) ** (2 * np.arange(1, n + 1))) / (1 - (1 - weight) ** (2))
        )

        return cls(
            ewma=ewma, target=target, lower_bound=lower_bound, upper_bound=upper_bound
        )
