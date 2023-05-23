Python API
**********

.. automodule:: benchml



Core
====

.. 
    automodule:: benchml.pipeline
        :members:

.. autoclass:: benchml.pipeline.Module

.. autoclass:: benchml.pipeline.Transform

Transforms are the nodes of the pipeline: They act on the data stream via calls to their .map and/or .fit methods. The results are then stored in their private stream and/or parameter object. An example for the constructor call that creates a new transform instance reads as follows:

.. code-block:: python

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

- The "args" dictionary supplies the parameters of the transformation, such as a cutoff, a convergence threshold, etc. These parameters should not be confused with the *output* parameters (which could, e.g., include fit coefficients or trained models) stored in the params() object of a transform.
- The "inputs" field contains links to the data stream of ancestral transforms on which the transformation acts. The address of the inputs is specified in the form <transform_tag>.<data_tag>. For example, "descriptor.x" points to the field "x" stored in the stream of the transform with name tag "descriptor". If the tag is prefixed with an underscore "\_" (such as in "predictor.\_model"), then the input is not read from the stream of the respective node, but its params object.

Implementing a new transform class
----------------------------------

There are three types of transforms: input, map, and fit+map transforms. Which type we are dealing with is determined by which methods (.\_feed, .\_map, .\_fit) a particular transform implements.

Input transforms
~~~~~~~~~~~~~~~~

Input transforms, such as ExtXyzInput, implement the .\_feed method that is called inside .open of a model (= pipeline):

.. code-block:: python

    stream = model.open(data) # < Internally this will call .feed on all
                              #   transforms that implement the ._feed method.

Below we show an example implementation for an input node (here: ExtXyzInput), where .feed is used to release "configs", "y" and "meta" into the data stream:

.. code-block:: python

    class ExtXyzInput(Transform):                 # < All transforms derive from <Transform>
        allow_stream = {'configs', 'y', 'meta'}   # < Fields permitted in the stream object
        stream_copy = ("meta",)                   # < See section on class attributes
        stream_samples = ("configs", "y")         # < See section on class attributes
        def _feed(self, data, stream):
            stream.put("configs", data)
            stream.put("y", data.y)
            stream.put("meta", data.meta)

Map transforms
~~~~~~~~~~~~~~

A map transform implements only .\_map (but not .\_fit). Most descriptors fall within this class of transforms, such as the RandomDescriptor class below:

.. code-block:: python

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
        def _map(self, inputs, stream):       # < The inputs dictionary comes preloaded with the appropriate data
            shape = (
              len(inputs["configs"]),
              self.args["dim"])
            X = np.random.uniform(
              self.args["xmin"],
              self.args["xmax"],
              size=shape)
            stream.put("X", X) # < The X matrix is stored in the active stream of the transform

Fit transforms
~~~~~~~~~~~~~~

Fit transforms implement .\_fit and .\_map: The former is called during the training stage within model.fit(stream). The fit stores its parameters in the transform.params() object, but may also access transform.stream(), e.g., to store predicted targets for the training set. The map operation reads model parameters from .params() (e.g. via self.params().get("coeffs")), and releases the mapped output into the stream. See below a wrapper around the Ridge predictor from sklearn:

.. code-block:: python

    class Ridge(Transform):
        default_args = { 'alpha': 1. }
        req_inputs = ('X', 'y')
        allow_params = {'model'}
        allow_stream = {'y'}
        def _fit(self, inputs, stream, params):
            model = sklearn.linear_model.Ridge(**self.args)
            model.fit(X=inputs["X"], y=inputs["y"])
            yp = model.predict(inputs["X"])
            params.put("model", model)
            stream.put("y", yp)
        def _map(self, inputs, stream):
            y = self.params().get("model").predict(inputs["X"])
            stream.put("y", y)

Transform class attributes
--------------------------

New transform classes may require us to update their class attributes in order to define default arguments, required inputs, or ensure correct handling of their data streams. The base Transform class lists the following class attributes:

.. code-block:: python

    class Transform(object):
        default_args = {}
        req_args = tuple()
        req_inputs = tuple()
        precompute = False
        allow_stream = {}
        allow_params = {}
        stream_copy = tuple()
        stream_samples = tuple()
        stream_kernel = tuple()

Computationally expensive transforms should typically set "precompute = True", which will add them to the list of transforms mapped during a call to model.precompute(stream). This will precompute the output for a specific data stream, and then only recompute values if the version hash of the stream changes (e.g., due to an args update of an ancestral transform).

For hyperoptimization, as well as benchmarking purposes, the stream attached to a transform needs to know how to split its data into a train and test partition. Consider e.g.,

.. code-block:: python

    stream = model.open(data)
    model.precompute(stream)
    stream_train, stream_test = stream.split(method="random", n_splits=5, train_fraction=0.7)
    model.fit(stream_train)

The stream_copy, stream_samples and stream_kernel attributes inform the stream how to adequately split its member data onto these partitions. For example, for ExtXyzInput, we have the following:

.. code-block:: python

    class ExtXyzInput(Transform):
        allow_stream = {'configs', 'y', 'meta'}
        stream_copy = ("meta",)
        stream_samples = ("configs", "y")

This will instruct the split operation to simply copy all the "meta" data to both stream_train and stream_test, whereas the "configs" and "y" data listed in stream_samples will be sliced (such as in configs_train = configs\[trainset\], configs_test = configs\[testset\]).

Finally, for a precomputed kernel object, this slicing operation differs qualitatively from slicing of, say, a design matrix, as this affects the two axes of the matrix in different way (e.g., K_train = K\[trainset\]\[:,trainset\], where K_test = K\[testset\]\[:,trainset\]. This is why the kernel matrix computed, e.g., by the KernelDot transform is listed in a dedicated stream_kernel attribute:

.. code-block:: python

    class KernelDot(Transform):
        default_args = {'power': 1}
        req_inputs = ('X',)
        allow_params = {'X'}
        allow_stream = {'K'}
        stream_kernel = ('K',)
        precompute = True

How to add a plugin
-------------------

New transforms can be defined either externally or internally. In the latter case, add a source file with the implementation to the benchml/plugins folder, and subsequently import that file in benchml/plugins/__init__.py. You can check that your transforms were successfully added using bin/bmark.py:

.. code-block:: console

    ./bin/bmark.py --list_transforms

