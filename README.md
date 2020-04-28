# Transforms

Transforms are the nodes of the pipeline: They act on the data stream via calls to their .map and/or .fit methods. The results are then stored in their private stream and/or parameter object. An example for the constructor call that creates a new transform instance reads as follows:
```python
trafo = RandomProjector(
  args={
    "cutoff": 0.1, 
    "epsilon": 0.01, 
    ...}, 
  inputs={
    "X": "descriptor.x", 
    "y": "input.y", 
    "M": "predictor._model",
    ...})
```
- The "args" dictionary supplies the parameters of the transformation, such as a cutoff, a convergence threshold, etc. These parameters should not be confused with the *output* parameters (which could, e.g., include fit coefficients or trained models) stored in the params() object of a transform.
- The "inputs" field contains links to the data stream of ancestral transforms on which the transformation acts. The address of the inputs is specified in the form <transform_tag>.<data_tag>. For example, "descriptor.x" points to the field "x" stored in the stream of the transform with name tag "descriptor". If the tag is prefixed with an underscore "_" (such as in "predictor._model"), then the input is not read from the stream of the respective node, but its params object.

## Defining a new transform class

```python
class FancyDescriptor(Transform):  # < All transforms derive from the <Transform> base class.
  default_args = {                 # < Default arguments (overwritten via the "args" dict in the constructor)
    'n_fancy': 2,
    'dim': None,
    'cutoff': None 
  }
  req_args = ('dim', 'cutoff')     # < Required arguments that need to be specified in the constructor "args"
  req_inputs = ('configs',)        # < Required inputs that need to be specified in the constructor "inputs"
  allow_stream = ('X',)            # < Fields permitted in the stream object of this transform. 
  def _map(self, inputs):
    configs = inputs['configs']    # < The inputs dictionary comes preloaded with the required data
    positions = configs.positions
    # <- ... compute X here ...
    self.stream().put("X", X)      # < The X matrix is stored in the active stream of the transform
```


# Construct a model
