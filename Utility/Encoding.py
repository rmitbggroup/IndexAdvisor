import Utility.PostgreSQL as pg


class Attribute:
    def __init__(self, table_name, attr_name, attr_id, is_key, data_type, is_origin, come_from):
        self.table_name = table_name
        self.attr_name = attr_name
        self.attr_id = attr_id
        self.is_key = is_key
        self.data_type = data_type
        self.is_origin = is_origin
        self.come_from = come_from


def encoding_schema(from_disk=False):
    tables_dict = dict()
    attributes_dict = dict()
    operation_dict = dict()
    operation_dict["eq"] = 0
    operation_dict["rg"] = 1
    table_order = dict()
    attrix2name = dict()

    if from_disk:
        pass
    else:
        pg_client = pg.PGHypo()
        tables = pg_client.get_tables('public')
        tables.sort()
        for i, table in enumerate(tables):
            attributes = pg_client.get_attributes(table, 'public')
            tables_dict[table] = len(attributes)
            table_order[table] = i
            _small_attr = dict()
            _ix2name = dict()
            for j, attribute in enumerate(attributes):
                info = attribute.split("#")
                a = info[0].find("key")
                attribute_instance = Attribute(table, info[0], j, a > -1, info[1], True, [])
                _small_attr[info[0]] = attribute_instance
                _ix2name[j] = info[0]
            attributes_dict[table] = _small_attr
            attrix2name[table] = _ix2name
        pg_client.close()
    encoding = dict()
    encoding['tb_list'] = tables
    encoding['tb_order'] = table_order
    encoding["tbl"] = tables_dict
    encoding["attr"] = attributes_dict
    encoding['ix2name'] = attrix2name
    encoding["op"] = operation_dict
    return encoding
