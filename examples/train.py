from sacred import Experiment
from mlflow_observer import MlflowObserver

from _paths import MY_TRACKING_URI

ex = Experiment('MyExperiment')
ex.observers.append(MlflowObserver(MY_TRACKING_URI))


@ex.config
def cfg():
    n_epochs = 30
    dataset = 'mnist'


@ex.automain
def train(dataset, n_epochs, _run):

    dataset = 'train-' + dataset
    for epoch in range(n_epochs):
        _run.log_scalar('accuracy', epoch*10 + 1, epoch)

    return True
