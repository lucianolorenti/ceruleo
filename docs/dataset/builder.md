# Builder

When building a PdM dataset you need the time series of the sensors of the machine and some indication of when the piece of equipment arrived to its end.


For this reason the DatasetBuilder class helps on this. It allow you to specify how your dataset is strucctured and split each run-to-failure cycle for posterior analysis.


## Failure modes
### Increasing feature
In scenarios where the dataset includes an increasing feature denoting the usage time of the item in question, it is possible to detect instances where a value at position 'i' is lower than the value at position 'i+1'. In such instances, we can establish that the item has been replaced. Consequently, we can determine the end of its lifespan as the last point within this increasing sequence.



```python
# mkdocs: render
# mkdocs: hidecode
import matplotlib.pyplot as plt
import numpy as np

xpoints= np.concatenate((
        np.linspace(0, 50, 50),
        np.linspace(0, 120, 120),
        np.linspace(0, 90, 90)
))
fig, ax = plt.subplots(2,1, sharex=True)

ax[0].plot(np.cos(xpoints) * np.sqrt(np.random.rand(len(xpoints))* xpoints))
ax[0].set_ylabel("Sensor value")
ax[0].set_title("Sensor value over time")

ax[1].set_title("Usage time")
ax[1].plot(xpoints)
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Usage time")
ax[1].axvline(x=50, color="red", label="End of life", linestyle="--")
ax[1].axvline(x=50+120, color="red", label="End of life", linestyle="--")
ax[1].axvline(x=50+120+90, color="red", label="End of life", linestyle="--")
ax[1].legend()

```

### Life identifier feature
In scenarios where the dataset includes a feature denoting each cycle ID, it is possible to detect samples for which the ID remains the same.



```python
# mkdocs: render
# mkdocs: hidecode
import matplotlib.pyplot as plt
import numpy as np

xpoints= np.concatenate((
        np.linspace(0, 50, 50),
        np.linspace(0, 120, 120),
        np.linspace(0, 90, 90)
))
fig, ax = plt.subplots(2,1, sharex=True)

ax[0].plot(np.cos(xpoints) * np.sqrt(np.random.rand(len(xpoints))* xpoints))
ax[0].set_ylabel("Sensor value")
ax[0].set_title("Sensor value over time")

ax[1].set_title("Cycle ID")
ax[1].plot(np.concatenate((np.ones(50), np.ones(120)*2, np.ones(90)*3)))
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Run to failure cycle ID")
ax[1].axvline(x=50, color="red", label="End of life", linestyle="--")
ax[1].axvline(x=50+120, color="red", label="End of life", linestyle="--")
ax[1].axvline(x=50+120+90, color="red", label="End of life", linestyle="--")
ax[1].legend()

```
### Cycle end identifier
In situations where a dataset contains a feature that indicats the end of a cycle, it is possible to segment data points based on this feature. Similar to detecting changes in ascending sequences, this process identifies transitions in the 'life end indicator' feature.


```python
# mkdocs: render
# mkdocs: hidecode
import matplotlib.pyplot as plt
import numpy as np

xpoints= np.concatenate((
        np.linspace(0, 50, 50),
        np.linspace(0, 120, 120),
        np.linspace(0, 90, 90)
))
fig, ax = plt.subplots(2,1, sharex=True)

ax[0].plot(np.cos(xpoints) * np.sqrt(np.random.rand(len(xpoints))* xpoints))
ax[0].set_ylabel("Sensor value")
ax[0].set_title("Sensor value over time")

ax[1].set_title("Cycle end indicator")
q = np.zeros(50+120+90)
q[50] = 1
q[50+120] = 1
q[50+120+90-1] = 1
ax[1].plot(q)
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Cycle end")

```

### Data and fault modes
In scenearios where the dataset is composed by two separates files: one with the sensor data and another with the fault data, it is possible to use the data and fault modes to split the run to failure cycles by combining both sources.  In that cases a datetime feature is required to align the data and fault modes.


```python
# mkdocs: render
# mkdocs: hidecode
import matplotlib.pyplot as plt
import numpy as np

xpoints= np.concatenate((
        np.linspace(0, 50, 50),
        np.linspace(0, 120, 120),
        np.linspace(0, 90, 90)
))
fig, ax = plt.subplots(1,2, gridspec_kw={'width_ratios': [8, 2]},figsize=(10, 5))

ax[0].plot(np.cos(xpoints) * np.sqrt(np.random.rand(len(xpoints))* xpoints))
ax[0].set_ylabel("Sensor value")
ax[0].set_title("Sensor value over time")
ax[0].axvline(x=50, color="red", label="Failure A", linestyle="--")
ax[0].axvline(x=50+120, color="purple", label="Failure B", linestyle="--")
ax[0].axvline(x=50+120+90, color="brown", label="Failure C", linestyle="--")
ax[0].legend()
fault_time = [50, 50+120, 50+120+90]
fault_modes = ['Failure A', 'Failure B', 'Failure C']

table_data = [['Time', 'Fault Mode']] + [[time, mode] for time, mode in zip(fault_time, fault_modes)]
table = ax[1].table(cellText=table_data, loc='center', cellLoc='center')
table.set_fontsize(14)  # Adjust font size if needed
ax[1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['bottom'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_xticks([])
ax[1].set_yticks([])


```

## Examples of usage

### Increasing feature

```python
df = pd.DataFrame(
    {
        "Cycle": list(range(0, 12, ))*2,
        "feature_a": list(range(12))*2,
        "feature_b": list(range(12, 24))*2,
    }
)
dataset = (
    DatasetBuilder()
    .set_splitting_method(IncreasingFeatureCycleSplitter("Cycle"))
    .set_rul_column_method(CycleRULColumn("Cycle"))
    .set_output_mode(InMemoryOutputMode())
    .build_from_df(df)
)
```


### Increasing with datetime based RUL feature

```python
df = pd.DataFrame(
    {
        "Cycle": list(range(0, 12, ))*2,
        "datetime": pd.date_range("2021-01-01", periods=24, freq="min").tolist(),
        "feature_a": list(range(12))*2,
        "feature_b": list(range(12, 24))*2,
    }
)
dataset = (
    DatasetBuilder()
    .set_splitting_method(IncreasingFeatureCycleSplitter("Cycle"))
    .set_rul_column_method(DatetimeRULColumn("datetime", "s"))
    .set_output_mode(InMemoryOutputMode())
    .build_from_df(df)
)
```

### Life identifier feature
```python
df = pd.DataFrame(
    {
        "life_id": [1]*12 + [2]*12,
        "datetime": pd.date_range("2021-01-01", periods=24, freq="min").tolist(),
        "feature_a": list(range(12))*2,
        "feature_b": list(range(12, 24))*2,
    }
)
dataset = (
    DatasetBuilder()
    .set_splitting_method(LifeIdCycleSplitter("life_id"))
    .set_rul_column_method(DatetimeRULColumn("datetime", "s"))
    .set_output_mode(InMemoryOutputMode())
    .build_from_df(df)
)
```

### Life end indicator feature
```python
df = pd.DataFrame(
    {
        "life_end": [0]*11 + [1] + [0]*11 + [1],
        "datetime": pd.date_range("2021-01-01", periods=24, freq="min").tolist(),
        "feature_a": list(range(12))*2,
        "feature_b": list(range(12, 24))*2,
    }
)
dataset = (
    DatasetBuilder()
    .set_splitting_method(LifeEndIndicatorCycleSplitter("life_end"))
    .set_rul_column_method(DatetimeRULColumn("datetime", "s"))
    .set_output_mode(InMemoryOutputMode())
    .build_from_df(df)
)
```

### Data and fault modes

```python
df = pd.DataFrame(
    {
        
        "datetime": pd.date_range("2021-01-01", periods=24, freq="min").tolist(),
        "feature_a": list(range(12))*2,
        "feature_b": list(range(12, 24))*2,
    }
)
failures = pd.DataFrame({
    "datetime_failure": [pd.Timestamp("2021-01-01 00:11:00"), pd.Timestamp("2021-01-01 00:23:00")],
    "failure_type": ["A", "B"]
})
dataset = (
    DatasetBuilder()
    .set_splitting_method(FailureDataCycleSplitter("datetime", "datetime_failure"))
    .set_rul_column_method(DatetimeRULColumn("datetime", "s"))
    .set_output_mode(InMemoryOutputMode())
    .build_from_df(df, failures)
)
```
"""
## Reference

### Dataset Builder 

::: ceruleo.dataset.builder 

### Cycles Splitter 

::: ceruleo.dataset.builder.cycles_splitter

