class TableInfo:
    def __init__(self, table_name):
        self.table_name = table_name
        self.required_columns = set()
        self.order = list()
        self.join_columns = set()
        self.eq_conditions = set()
        self.range_conditions = set()
        self.or_conditions = list()