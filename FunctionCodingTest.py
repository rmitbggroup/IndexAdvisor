"""import Utility.PostgreSQL as pg
tables = pg.get_tables('public')
for i, table in enumerate(tables):
    attributes = pg.get_attributes(table, 'public')
    new_attribures = list()
    print(i)
    for j, attribute in enumerate(attributes):
        a = attribute.find("key")
        if a > -1:
            new_attribures.append(attribute+",y")
        else:
            new_attribures.append(attribute + ",x")
    info = table+":" + ";".join(new_attribures)
    print(info)"""