## Transforms

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
- The "inputs" field contains links to the data stream of ancestral transforms on which the transformation acts. The address of the inputs is specified in the form <transform_tag>.<data_tag>. For example, "descriptor.x" points to the field "x" stored in the stream of the transform with name tag "descriptor". If the tag is prefixed with an underscore "\_" (such as in "predictor.\_model"), then the input is not read from the stream of the respective node, but its params object.

### Implementing a new transform class

There are three types of transforms: Input, map, and fit+map transforms. Input transforms, such as ExtXyzInput, implement a .\_feed method that is called inside the .open method of a model (= pipeline):
```python
stream = model.open(data) # < Internally this will call .feed(data) on all 
                          #   transforms that implement the ._feed method.
```
Below we show an example implementation for an input node (here: ExtXyzInput), where the .feed method is used to release "configs", "y" and "meta" into the data stream:
```python
class ExtXyzInput(Transform):                 # < All transforms derive from <Transform>
    allow_stream = {'configs', 'y', 'meta'}   # < Fields permitted in the stream object
    stream_copy = ("meta",)                   # < See section on class attributes
    stream_samples = ("configs", "y")         # < See section on class attributes
    def _feed(self, data):
        self.stream().put("configs", data)
        self.stream().put("y", data.y)
        self.stream().put("meta", data.meta)
```

A map transform implements only .\_map (but neither .\_feed nor .\_fit). Most descriptors fall within this class of transforms, such as the RandomDescriptor class below:
```python
class RandomDescriptor(Transform):
    default_args = {
      'xmin': -1.,
      'xmax': +1.,
      'dim': None
    }
    req_args = ('dim',)           # < Required fields to be specified in the constructor "args"
    req_inputs = ('configs',)     # < Required inputs to be specified in the constructor "inputs"
    allow_stream = {'X'}
    stream_samples = ('X',)
    precompute = True
    def _map(self, inputs):       # < The inputs dictionary comes preloaded with the appropriate data
        shape = (
          len(inputs["configs"]), 
          self.args["dim"])
        X = np.random.uniform(
          self.args["xmin"], 
          self.args["xmax"], 
          size=shape)
        self.stream().put("X", X) # < The X matrix is stored in the active stream of the transform
```

# Construct a model
