# Run-to-failure cycles duration

Entender la duración de los ciclos de fallo es indispensable  


``` py
from ceruleo.dataset.catalog.CMAPSS import CMAPSSDataset
from ceruleo.graphics.duration import lives_duration_histogram

train_dataset = CMAPSSDataset(train=True, models='FD001')
validation_dataset = CMAPSSDataset(train=False, models='FD001')
lives_duration_histogram([train_dataset,validation_dataset], 
                         label=['Train dataset','Validation dataset'],
                         xlabel='Unit Cycles', 
                         alpha=0.7,
                         units='cycles',
                         figsize=(17, 5));
``` 
![Duration](/img/duration_histogram.png){ align=left }

## Reference 

::: ceruleo.graphics.duration
    options:
      members:
        - durations_boxplot
        - durations_histogram