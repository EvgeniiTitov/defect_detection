from abc import ABC, abstractmethod, abstractproperty


class AbstractWorker(ABC):

    def __init__(self, q_in, q_out):
        self._is_alive = False
        self._q_in = q_in
        self._q_out = q_out

    @abstractmethod
    def kill_worker(self):
        pass

    @abstractmethod
    def run_worker(self):
        pass
