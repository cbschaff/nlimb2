"""A decorator for computation in subprocesses

@subproc_worker
class MyClass():
    def __init__(self, x):
        self.x = x

    def add_to_x(self, y):
        return self.x + y

my_obj = MyClass(x=5)              # Class is instantiated in a subprocess
job = my_obj.add_to_x(5)           # methods are run in the subprocess and return a Job object
print(job.job_id, job.cmd, job.args, job.kwargs)  # "(0, 'add_to_x', (5,), {}"
print(job.results)                 # Blocks until results are ready. Prints 10
my_obj.close()                     # Closes the subprocess. This also is called in the destructor

"""
import multiprocessing as mp
import inspect
import pickle
import cloudpickle


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = cloudpickle.loads(ob)


def worker(remote, cls, init_args, init_kwargs):
    try:
        obj = cls.x(*init_args.x, **init_kwargs.x)
        while True:
            cmd, data = remote.recv()
            if cmd == 'close':
                if hasattr(obj, 'close'):
                    obj.close()
                remote.close()
                break
            if hasattr(obj, cmd):
                args, kwargs = data.x
                out = getattr(obj, cmd)(*args, **kwargs)
                remote.send(CloudpickleWrapper(out))
            else:
                raise ValueError('Worker got unknown command {cmd}')
    except KeyboardInterrupt:
        remote.close()


class JobLedger():
    def __init__(self, remote):
        self._outputs = {}
        self._count_finished = 0
        self._count_started = 0
        self._remote = remote

    def add_job(self, cmd, args, kwargs):
        self.check_for_results()
        data = CloudpickleWrapper((args, kwargs))
        self._remote.send((cmd, data))
        jid = self._count_started
        self._outputs[jid] = None
        self._count_started = self._count_started + 1
        return Job(cmd, args, kwargs, jid, ledger=self)

    def _add_result(self):
        data = self._remote.recv()
        if self._count_finished in self._outputs:
            self._outputs[self._count_finished] = data.x
        self._count_finished += 1

    def get_results(self, job_id):
        while self._count_finished <= job_id and not self._remote.closed:
            self._add_result()
        if job_id not in self._outputs:
            raise ValueError(f"Output of job {job_id} can't be found. This is likely because it "
                              "has been removed from the ledger.")
        return self._outputs[job_id]

    def delete_results(self, job_id):
        self.check_for_results()
        if job_id in self._outputs:
            del self._outputs[job_id]

    def check_for_results(self):
        if self._remote.closed:
            return
        while self._remote.poll():
            self._add_result()

    def is_complete(self, job_id):
        self.check_for_results()
        return self._count_finished > job_id


class Job():
    def __init__(self, cmd, args, kwargs, job_id, ledger):
        self.cmd = cmd
        self.args = args
        self.kwargs = kwargs
        self.job_id = job_id
        self._ledger = ledger

    def is_finished(self):
        return self._ledger.is_complete(self.job_id)

    @property
    def results(self):
        return self._ledger.get_results(self.job_id)

    def join(self):
        self._ledger.get_results(self.job_id)

    def __del__(self):
        self._ledger.delete_results(self.job_id)


def _subproc_decorator(cls, ctx, daemon):

    class RemoteWorker():
        def __init__(self, *args, **kwargs):
            self._closed = False
            self.ctx = mp.get_context(ctx)
            self.remote, self.child = self.ctx.Pipe()
            self.proc = self.ctx.Process(target=worker,
                                         args=(self.child, CloudpickleWrapper(cls),
                                               CloudpickleWrapper(args),
                                               CloudpickleWrapper(kwargs)))
            self.proc.daemon = daemon
            self.proc.start()
            self._ledger = JobLedger(self.remote)

        def close(self):
            if not self._closed:
                self.remote.send(('close', {}))
                self.remote.close()
                self._closed = True

        def __del__(self):
            self.close()

    def _add_command(name):
        def remote_fn(self, *args, **kwargs):
            return self._ledger.add_job(name, args, kwargs)
        setattr(RemoteWorker, name, remote_fn)

    for name, _ in inspect.getmembers(cls, inspect.isfunction):
        if name[0] == '_' or name == 'close':
            continue
        _add_command(name)

    return RemoteWorker


def subproc_worker(cls=None, ctx='fork', daemon=True):
    if cls is None:
        return lambda cls: _subproc_decorator(cls, ctx, daemon)
    else:
        return _subproc_decorator(cls, ctx, daemon)


if __name__ == '__main__':
    @subproc_worker
    class MyClass():
        def __init__(self, x):
            self.x = x

        def add_to_x(self, y):
            return self.x + y

    my_obj = MyClass(x=5)              # Class is instantiated in a subprocess
    job = my_obj.add_to_x(5)           # methods are run in the subprocess and return a Job object
    print(job.job_id, job.cmd, job.args, job.kwargs)  # "(0, 'add_to_x', (5,), {}"
    print(job.results)                 # Blocks until results are ready. Prints 10
    my_obj.close()                     # Closes the subprocess. This also is called in the destructor
