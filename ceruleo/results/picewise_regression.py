from typing import List, Tuple

import numpy as np


class Segment:
    def __init__(self, initial_point:Tuple[float, float], not_increasing:bool = False):
        self.n = 1
        self.initial = initial_point
        self.xx = 0
        self.xy = 0
        self.yy = 0
        self.B = 0
        self.segment_error = []
        self.std_deviation = 2
        self.not_increasing = not_increasing

    def compute_endpoint(self, current_t: float):
        # Predict with previous point
        t = current_t - self.initial[0]
        B = self.B 
        if self.not_increasing:
            B = min(B, 0)
        hat_s = self.initial[1] + (B * (t))
        self.final = (current_t, hat_s)

    def can_add(self, current_t: float, value: float) -> bool:
        n = self.n + 1
        y = value - self.initial[1]
        t = current_t - self.initial[0]
        xx = self.xx + (((t * t) - self.xx)) / n
        xy = self.xy + (((t * y) - self.xy)) / n
        yy = self.yy + (((y * y) - self.yy)) / n
        if xx > 0:
            B = xy / xx
        else:
            B = 0
        SSR = (yy - B * xy) * self.n
        new_segment = False
        if self.n > 15:
            mean_error = np.mean(self.segment_error)
            std_error = np.std(self.segment_error)
            new_segment = SSR > mean_error + 1.5 *std_error
        else:
            new_segment = False
        return not new_segment

    def add(self, current_t: float, value: float):
        self.n = self.n + 1
        y = value - self.initial[1]
        t = current_t - self.initial[0]

        self.xx = self.xx + (((t * t) - self.xx)) / self.n
        self.xy = self.xy + (((t * y) - self.xy)) / self.n
        self.yy = self.yy + (((y * y) - self.yy)) / self.n
        if self.xx > 0:
            self.B = self.xy / self.xx
        else:
            self.B = 0
        SSR = (self.yy - self.B * self.xy) * self.n
        self.segment_error.append(SSR)
        self.compute_endpoint(current_t)


class PiecewesieLinearFunction:
    """Function defined picewise with linear segments

    Parameters
    ----------
    segments: List[Segment]
        List of segments that compose the function
    """

    def __init__(self, segments: List[Segment]):
        self.parameters = []
        self.limits = []
        for s in segments:
            t1, y1 = s.initial
            t2, y2 = s.final
            if (t2 - t1) == 0:
                m = 0
            else:
                m = (y2 - y1) / (t2 - t1)
            b = y1 - m * t1
            self.parameters.append((m, b))
            self.limits.append((t1, t2))


    def predict_line(self, x_values):
        return [self.predict(x) for x in x_values]

    def predict(self, x):
        for i, (l1, l2) in enumerate(self.limits):
            if x >= l1 and x <= l2:
                m, b = self.parameters[i]

                return m * x + b
        m, b = self.parameters[-1]
        return m * x + b

    def zero(self):
        for i, ((l1, l2), (m, b)) in enumerate(zip(self.limits, self.parameters)):
            if m == 0:
                continue
            x_0 = -b / m
            if (x_0 >= l1) and (x_0 <= l2):
                return x_0       
        
        m, b = self.parameters[-1]
        while m == 0:
            return 0
        return -b / m


class PiecewiseLinearRegression:
    """Perform a picewise linear regression

    The method is the one presented in:

    Time and Memory Efficient Online Piecewise Linear Approximation of Sensor Signals
    Florian Grützmacher, Benjamin Beichler, Albert Hein,Dand Thomas Kirste and Christian Haubelt

    """

    def __init__(self, not_increasing:bool=False):
        self.segments = []
        self.not_increasing = not_increasing

    def add_point(
        self,
        t: float,
        s: float,
    ):
        """Add a new point to the regression

        Parameters
        ----------
        t : float
            x component
        s : float
            y component
        """
        if len(self.segments) == 0:
            self.segments.append(Segment((t, s), not_increasing=self.not_increasing))
            return

        if self.segments[-1].can_add(t, s):
            self.segments[-1].add(t, s)
        else:
            self.segments.append(Segment(self.segments[-1].final, not_increasing=self.not_increasing))
            self.segments[-1].add(t, s)

    def finish(self) -> PiecewesieLinearFunction:
        """Complete last unfinished segment and return the model computed

        Returns
        -------
        PiecewesieLinearFunction
            The picewise linear model fitted
        """
        return PiecewesieLinearFunction(self.segments)
