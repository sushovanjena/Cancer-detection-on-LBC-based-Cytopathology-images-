from multiprocessing import Process, Queue


class MemoryManager:
    def __init__(self, mem_limit):
        self.__mem_limit = mem_limit
        self.__proc_in_q = Queue()
        self.__proc_out_q = Queue()
        self.__is_mem_acquired = False
        self.mem_acq_proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.release_mem()
        return self

    def _tf_mem_process(self, tf_proc_in_q, tf_proc_out_q, memory=500):
        import cupy as cp
        while True:
            val = tf_proc_in_q.get()
            if val is None:
                return

            # noinspection PyBroadException
            try:
                a = cp.random.randint(0, 256, (6000, 6000))
                tf_proc_out_q.put(1)
            except Exception as e:
                tf_proc_out_q.put(0)

    def acquire_mem(self):
        if not self.__is_mem_acquired:
            self.mem_acq_proc = Process(target=self._tf_mem_process, args=(self.__proc_in_q, self.__proc_out_q, self.__mem_limit))
            self.mem_acq_proc.start()
            self.__is_mem_acquired = True
            self.__proc_in_q.put(self.__is_mem_acquired)
            status = self.__proc_out_q.get()
            if status == 0:
                raise RuntimeError("tf mem acq failed")

    def release_mem(self):
        if self.__is_mem_acquired:
            self.__proc_in_q.put(None)
            self.mem_acq_proc.join()
            self.__is_mem_acquired = False



