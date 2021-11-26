


::: ceruleo.iterators.iterators
    handler: python
    options:
      members:
        - WindowedDatasetIterator

## Iteration Type

::: ceruleo.iterators.iterators.IterationType
    handler: python
    show_root_heading: true



## Sample weights

The Sample Weight type is a callable with the following signature

    fun(y, i:int, metadata)

Given the target and the sample index `i` it returns the sample weight for sample `i`.
There area few callable classes already made with standard sample weight schemes in PdM.


::: ceruleo.iterators.sample_weight.SampleWeight
    options:
      heading_level: 3
      show_root_heading: true
      
::: ceruleo.iterators.sample_weight.AbstractSampleWeights
    options:
      heading_level: 3
      show_root_heading: true


::: ceruleo.iterators.sample_weight.NotWeighted
    options:
      heading_level: 3
      show_root_heading: true


::: ceruleo.iterators.sample_weight.RULInverseWeighted
    options:
      heading_level: 3
      show_root_heading: true


::: ceruleo.iterators.sample_weight.InverseToLengthWeighted
    options:
      heading_level: 3
      show_root_heading: true









## Relative positioning

Sometimes is useful to iterate the run-to-failure cycle starting or ending at some specifig indices.
These classes allow to specify relative positions to start or end the iteration.


::: ceruleo.iterators.iterators.RelativePosition
    options:
      heading_level: 3
      show_root_heading: true

::: ceruleo.iterators.iterators.RelativeToStart
    options:
      heading_level: 3
      show_root_heading: true

::: ceruleo.iterators.iterators.RelativeToEnd
    options:
      heading_level: 3
      show_root_heading: true
