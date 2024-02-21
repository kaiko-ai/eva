"""Multiprocessing utilities."""

import multiprocessing
import sys
import traceback


class Process(multiprocessing.Process):
    """multiprocessing.Process wrapper with logic to propagate exceptions to the parent process.

    Source: https://stackoverflow.com/a/33599967/4992248
    """

    def __init__(self, *args, **kwargs):
        """Initialize the process."""
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._parent_conn, self._child_conn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        """Run the process."""
        try:
            multiprocessing.Process.run(self)
            self._child_conn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))

    @property
    def exception(self):
        """Property that contains exception information from the process."""
        if self._parent_conn.poll():
            self._exception = self._parent_conn.recv()
        return self._exception

    def check_exceptions(self):
        """Check for exception propagate it to the parent process."""
        if not self.is_alive():
            if self.exception:
                error, traceback = self.exception
                sys.stderr.write(traceback + "\n")
                raise error
