mlflow-observer
===============
Observe your `sacred <https://github.com/IDSIA/sacred>`_ experiments with `mlflow <https://github.com/mlflow/mlflow>`_.


Writing experiments with ``sacred`` is great.

``mlflow`` provides a nice UI that can be used to get a quick overview of your runs and analyze the results.


Usage
-----
In your code, add the observer:

.. code-block:: python

    from sacred import Experiment
    from mlflow_observer import MlflowObserver

    from _paths import MY_TRACKING_URI

    ex = Experiment('MyExperiment')
    ex.observers.append(MlflowObserver(MY_TRACKING_URI))


In the commandline, you can pass a run name through sacred's comment flag:

.. code-block:: bash

  python train.py -c "My sacred run"

 
Otherwise the run name will be of the form ``run_[datetime]``.
