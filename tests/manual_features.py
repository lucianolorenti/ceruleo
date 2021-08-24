import scipy.stats
import pandas as pd
import numpy as np

def manual_expanding(df: pd.DataFrame, min_points:int= 1):
    to_compute = ['kurtosis', 'skewness', 'max', 'min', 'std', 'peak', 'impulse',
            'clearance', 'rms', 'shape', 'crest']
    dfs = []
    for c in df.columns:
        d = []
        for i in range(min_points-1):
            d.append([np.nan for f in to_compute])
        for end in range(min_points, df.shape[0]+1):
            data = df[c].iloc[:end]
            row = [manual_features(data, f) for f in to_compute]
            d.append(row)
        dfs.append(pd.DataFrame(d, columns=[f'{c}_{f}' for f in to_compute]))
    return pd.concat(dfs, axis=1)



def kurtosis(s: pd.Series) -> float:
    return scipy.stats.kurtosis(s.values, bias=False)


def skewness(s: pd.Series) -> float:
    return scipy.stats.skew(s.values, bias=False)


def max(s: pd.Series) -> float:
    return np.max(s.values)


def min(s: pd.Series) -> float:
    return np.min(s.values)


def std(s: pd.Series) -> float:
    return np.std(s.values, ddof=1)


def peak(s: pd.Series) -> float:
    return max(s) - min(s)


def impulse(s: pd.Series) -> float:
    return peak(s) / np.mean(np.abs(s))


def clearance(s: pd.Series) -> float:
    return peak(s) / (np.mean(np.sqrt(np.abs(s)))**2)


def rms(s: pd.Series) -> float:
    return np.sqrt(np.mean(s**2))


def shape(s: pd.Series) -> float:
    return rms(s) / np.mean(np.abs(s))


def crest(s: pd.Series) -> float:
    return peak(s) / rms(s)


feature_functions = {
    'crest': crest,
    'shape': shape,
    'rms': rms,
    'clearance': clearance,
    'impulse': impulse,
    'peak': peak,
    'std': std,
    'min': min,
    'max': max,
    'skewness': skewness,
    'kurtosis': kurtosis
}


def manual_features(s: pd.Series, name: str):
    return feature_functions[name](s)