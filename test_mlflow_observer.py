import os
import shutil
import pickle
import pytest
import tempfile

from sacred import Experiment
from mlflow_observer import MlflowObserver


ex = Experiment('MyExperiment')

METRIC_NAME = 'accuracy'
ARTIFACT_NAME = 'model.pkl'
ARTIFACTS_SUBDIR = 'my_models'
ARTIFACT_NAME_FMT = 'model_{}.pkl'


@ex.config
def cfg():
    n_epochs = 10
    dataset = 'mnist'


@ex.automain
def train(dataset, n_epochs, _run):

    # configuration is stored during the 'started_event'

    dataset = 'train-' + dataset
    for epoch in range(n_epochs):
        # store a metric
        _run.log_scalar(METRIC_NAME, epoch*10 + 1, epoch)

    run_dir = tempfile.mkdtemp()

    # store an artifact
    path = os.path.join(run_dir, ARTIFACT_NAME)
    model = {'w': 231.5, 'b': 43.12}

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    _run.add_artifact(path)

    # store a directory of artifacts
    models_dir = os.path.join(run_dir, ARTIFACTS_SUBDIR)
    os.mkdir(models_dir)
    for i in range(3):
        path = os.path.join(models_dir, ARTIFACT_NAME_FMT.format(i))
        model = {'w': 231.5 + i, 'b': 43.12 + i}

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    _run.add_artifact(models_dir)

    return True


@pytest.mark.parametrize("tracking_uri", ["mlruns", "my_mlruns_dir"])
@pytest.mark.parametrize("run_name", [None, "My sacred run"])
def test_file_storage(tracking_uri, run_name):
    obs = MlflowObserver(tracking_uri=tracking_uri)
    ex.observers = [obs]

    meta_info = None if run_name is None else {'comment': run_name}
    run = ex.run(meta_info=meta_info)

    assert os.path.exists(tracking_uri)

    experiment_id = '1'
    run_id = run._id
    run_uri = os.path.join(tracking_uri, experiment_id, run_id)
    assert os.path.exists(run_uri)

    # artifacts
    artifacts_uri = os.path.join(run_uri, 'artifacts')
    assert os.path.exists(artifacts_uri)

    model_uri = os.path.join(artifacts_uri, ARTIFACT_NAME)
    assert os.path.exists(model_uri)

    models_uri = os.path.join(artifacts_uri, ARTIFACTS_SUBDIR)
    assert os.path.isdir(models_uri)

    # metrics
    metrics_uri = os.path.join(run_uri, 'metrics')
    assert os.path.exists(metrics_uri)

    accuracy_uri = os.path.join(metrics_uri, METRIC_NAME)
    assert os.path.exists(accuracy_uri)

    # configuration
    config_uri = os.path.join(run_uri, 'params')
    assert os.path.exists(config_uri)

    for key, value in run.config.items():
        kv_path = os.path.join(config_uri, key)
        assert os.path.exists(kv_path)
        assert open(kv_path, 'r').read() == str(value)

    # tags - run name
    tags_uri = os.path.join(run_uri, 'tags')
    assert os.path.exists(tags_uri)

    run_name_uri = os.path.join(tags_uri, 'mlflow.runName')
    assert os.path.exists(run_name_uri)

    stored_run_name = open(run_name_uri, 'r').read()

    if run_name is not None:
        assert stored_run_name == run_name
    else:
        assert stored_run_name.startswith("run_")

    # delete files
    shutil.rmtree(tracking_uri)
