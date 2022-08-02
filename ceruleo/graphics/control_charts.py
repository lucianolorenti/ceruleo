from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def format_relative_time(x, pos):
    return timedelta(seconds=x)


def format_xticks(ax):
    formatter = FuncFormatter(format_relative_time)
    ax.xaxis.set_major_formatter(formatter)


def format_xaxis_time(ax):
    format_xticks(ax)
    ax.set_xlabel('Relative time from the beggining')


def ewma(feat, lambda_=0.5):
    x = np.nanmean(feat)
    s = np.sqrt(lambda_ / (2-lambda_))*np.nanstd(feat)
    UCL = x + 3*s
    LCL = x - 3*s
    return (LCL, x, UCL)


def plot_ewma(df, x_feat_name, y_feat_name, lambda_=0.5, ax=None, description='', units=None):
    plot_ewma_(df[x_feat_name],
               df[y_feat_name],
               description,
               units,
               lambda_=lambda_,
               ax=ax)


def plot_ewma_(time, feat, description, units, lambda_=0.5, ax=None, color=None):
    LCL, mean, UCL = ewma(feat, lambda_=lambda_)
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 4))

    ax.plot(time, feat, color='blue')
    ax.axhline(
        y=LCL,
        label=f'LCL={np.round(LCL, 2)} | $\\lambda = {lambda_}$',
        color=color)
    ax.axhline(y=mean)
    ax.axhline(
        y=UCL,
        label=f'UCL={np.round(UCL, 2)} | $\\lambda = {lambda_}$',
        color=color)
    ax.set_title('EWMA Chart ' + description)
    # format_xaxis_time(ax)
    ax.set_ylabel(description + (f' [{units}]' if units else ''))
    # format_xaxis_time(ax)
    ax.legend()

    return LCL, mean, UCL, ax
