import sys
import io
import contextlib
import time

def capture_output(func):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func()
        sys.stdout.flush()  # Force flushing to ensure immediate capture
    return buffer.getvalue().strip()
