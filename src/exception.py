import sys

class Custom_Exception(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors
        # Extract traceback details if you want to use them
        _, _, exc_tb = sys.exc_info()
        if exc_tb is not None:
            self.file_name = exc_tb.tb_frame.f_code.co_filename
            self.line_number = exc_tb.tb_lineno
        else:
            self.file_name = None
            self.line_number = None

    def __str__(self):
        if self.file_name and self.line_number:
            return f"{super().__str__()} (File: {self.file_name}, Line: {self.line_number})"
        else:
            return super().__str__()