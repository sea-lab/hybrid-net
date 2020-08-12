import pandas

pandas.set_option('display.max_columns', None)  # show all columns
pandas.options.display.float_format = '{:,.2f}'.format


class PandasFloatFormatter:
    def __init__(self, new_format: str):
        self.new_format = new_format.format
        self.old_format = None

    def __enter__(self):
        self.old_format = pandas.options.display.float_format
        pandas.options.display.float_format = self.new_format

    def __exit__(self, exc_type, exc_val, exc_tb):
        pandas.options.display.float_format = self.old_format
