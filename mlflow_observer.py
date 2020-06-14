import os
import datetime
from sacred.observers import RunObserver

from mlflow.entities import RunStatus, Metric
from mlflow.tracking import MlflowClient
from mlflow import (
    set_tracking_uri,
    set_experiment,
    start_run,
    end_run,
    log_params
)


__version__ = "0.0.1"


class MlflowObserver(RunObserver):
    """Observe configuration, metrics and artifacts with mlflow.

    Parameters
    ----------
    tracking_uri : str, optional
        Either a directory path or a database url. If None (default),
        an 'mlruns' directory will be created and used in the local
        file system.

    time_fmt : str, optional
        The run name defaults to a string of the form "run_[timestamp]".
        The timestamp is formatted according to time_fmt. If a comment is
        passed through sacred, it will be used as the run's name instead.

    Attributes
    ----------

    _client : MlflowClient
        a client connecting to the (remote) store.

    _run_id : str
        the generated id for the run.

    """

    priority = 0

    def __init__(self, tracking_uri=None, time_fmt="%b%d_%H-%M-%S"):
        self.tracking_uri = tracking_uri
        self.time_fmt = time_fmt

        self._client = MlflowClient(self.tracking_uri)
        self._run_id = None

    def queued_event(self, ex_info, command, host_info, queue_time, config,
                     meta_info, _id):
        pass

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        """Start the mlflow run and return its id"""

        set_tracking_uri(self.tracking_uri)
        set_experiment(ex_info['name'])

        # a user can pass a run name through sacred's comment flag (e.g. -c first_run)
        # otherwise run name defaults to the current timestamp
        name = meta_info.get('comment')
        if name is None:
            now = datetime.datetime.now().strftime(self.time_fmt)
            name = f"run_{now}"

        run = start_run(run_name=name)

        log_params(config)

        self._run_id = run.info._run_id

        return self._run_id

    def heartbeat_event(self, info, captured_out, beat_time, result):
        pass

    def completed_event(self, stop_time, result):
        end_run(status=RunStatus.to_string(RunStatus.FINISHED))

    def interrupted_event(self, interrupt_time, status):
        end_run(status=RunStatus.to_string(RunStatus.KILLED))

    def failed_event(self, fail_time, fail_trace):
        end_run(status=RunStatus.to_string(RunStatus.FAILED))

    def resource_event(self, filename):
        pass

    def log_metrics(self, metrics_by_name, info):
        """Store new measurements to the database.

        This is called in every heartbeat event (and when the run ends).

        sacred stores timestamps as datetime.datetime.utcnow() objects
        mlflow stores timestamps as int(time.time()) objects

        To convert sacred timestamps to mlflow timestamps:

            mlflow_timestamp = int(sacred_timestamp.timestamp() * 1000)

        """
        for name, metric_dict in metrics_by_name.items():
            steps = metric_dict['steps']
            values = metric_dict['values']
            timestamps = metric_dict['timestamps']
            batch = [Metric(name, v, int(t.timestamp() * 1000), step)
                     for v, t, step in zip(values, timestamps, steps)]

            self._client.log_batch(self._run_id, metrics=batch)

    def artifact_event(self, name, filename, metadata=None, content_type=None):
        """mlflow can store whole directories, not just single files"""

        if os.path.isdir(filename):
            # store files found under the local directory
            # in a remote uri under the directory's name
            dir_name = os.path.basename(filename)
            self._client.log_artifacts(self._run_id, local_dir=filename, artifact_path=dir_name)
        else:
            self._client.log_artifact(self._run_id, local_path=filename)
