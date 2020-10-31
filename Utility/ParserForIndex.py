import itertools


class Condition:
    def __init__(self, table_name, _type, column, value_type, otherside):
        # type join/=/>/</like;
        self.table_name = table_name
        self.column = column
        self.type = _type
        self.value_type = value_type
        self.otherside = otherside
        self.selectivity = 0.0


class Table:
    def __init__(self, table_name, alias_name, is_origin):
        self.alias_name = alias_name
        self.table_name = table_name
        self.columns = dict()
        self.used_columns = set()
        self.conditions = []
        self.join_conditions = []
        self.group = dict()
        self.order = dict()
        self.is_origin = is_origin

    def set_columns(self, columns):
        self.columns = columns

    def add_used_column(self, column):
        self.used_columns.add(column)


class Parser:
    def __init__(self, db_info):
        self.db_info = db_info
        self.table_info = dict()
        self.support_op = {'=': 0, '>': 1, '<': 2, '>=': 3, '<=': 4}
        self.index_candidates = set()

    def reset_candidates(self):
        self.index_candidates = set()

    def gain_candidates(self):
        for tb in self.table_info:
            tb_info = self.table_info[tb]
            if tb_info.table_name != tb:
                tb = tb_info.table_name

            # handle join
            joined_table = set()
            joined_id = set()
            joined_index = set()
            if len(tb_info.join_conditions) > 0:
                for cond in tb_info.join_conditions:
                    _other_table = cond.otherside.split('#@#')[0]
                    if _other_table != tb and not (_other_table+"#"+cond.column in joined_id):
                        self.index_candidates.add(tb+"#"+cond.column)
                        joined_index.add(tb+"#"+cond.column)
                        if _other_table in joined_table:
                            # multiple column join
                            _p_j_col1 = [cond.column]
                            _p_j_col2 = []
                            for _p in joined_id:
                                _t_in_p = _p.split('#')[0]
                                if _t_in_p == _other_table:
                                    _p_j_col1.append(_p.split('#')[1])
                                    _p_j_col2.append(_p.split('#')[1])

                            # permutation
                            c1 = set()
                            c2 = set()
                            for _n_attr in range(2, len(_p_j_col1)+1):
                                permutations = list(itertools.permutations(_p_j_col1, _n_attr))
                                for permutation in permutations:
                                    c1.add(permutation)
                            for _n_attr in range(2, len(_p_j_col2)+1):
                                permutations = list(itertools.permutations(_p_j_col2, _n_attr))
                                for permutation in permutations:
                                    c2.add(permutation)
                            _f_c = c1 - c2
                            f_c = set()
                            for _o_f_c in _f_c:
                                a_in_os = []
                                for a_in_o in _o_f_c:
                                    a_in_os.append(a_in_o)
                                f_c.add(tb+"#"+','.join(a_in_os))
                            # ERROR
                            self.index_candidates = self.index_candidates | f_c
                            joined_index = joined_index | f_c
                        joined_table.add(_other_table)
                        joined_id.add(_other_table+"#"+cond.column)

            # handle conditions
            mix_indexes = set() | joined_index
            if len(tb_info.conditions) > 0:
                e_cols = set()
                ne_cols = set()

                # eq condition
                for c_cond in tb_info.conditions:
                    self.index_candidates.add(tb + '#' + c_cond.column)
                    if c_cond.type == '=':
                        e_cols.add(c_cond.column)

                    else:
                        ne_cols.add(c_cond.column)

                # permutation
                for _j_idx in joined_index:
                    _n_cols = set(_j_idx.split('#')[1].split(','))
                    _n_cols = _n_cols | e_cols
                    for _n_attr in range(2, len(_n_cols) + 1):
                        permutations = list(itertools.permutations(_n_cols, _n_attr))
                        temp = set()
                        for permutation in permutations:
                            a_in_ps = []
                            for a_in_p in permutation:
                                a_in_ps.append(a_in_p)
                            temp.add(tb + "#" + ','.join(a_in_ps))
                        n_temp = temp - mix_indexes
                        mix_indexes = mix_indexes | n_temp

                # non-eq condition
                temp = set()
                for _m_idx in mix_indexes:
                    for _ne_col in ne_cols:
                        _m_ne_col = _m_idx + "," + _ne_col
                        temp.add(_m_ne_col)
                temp = mix_indexes | temp
                new_added = set()
                if len(tb_info.used_columns) <=3:
                    for _f2_idx in temp:
                        cols = set(_f2_idx.split('#')[1].split(','))
                        _f_copy = _f2_idx
                        is_new = False
                        for u_col in tb_info.used_columns:
                            if not (u_col in cols):
                                _f_copy = _f_copy + ',' + u_col
                                is_new = True
                        if is_new:
                            new_added.add(_f_copy)
                self.index_candidates = self.index_candidates | temp | new_added

    def parse_stmt(self, stmt):
        self.table_info = dict()
        select_stmt = stmt['SelectStmt']
        self.parse_select(select_stmt)
        return

    def parse_range_var(self, range_var):
        table_name = range_var['relname']
        alias_name = table_name
        if 'alias' in range_var.keys():
            alias_name = range_var['alias']['Alias']['aliasname']

        if alias_name in self.table_info.keys():
            # subquery/sublink
            return
        self.table_info[alias_name] = Table(table_name, alias_name, True)
        self.table_info[alias_name].set_columns(self.db_info[table_name])

    def parse_range_subselect(self, subselect):
        table_name = subselect['alias']['Alias']['aliasname']
        alias_name = table_name
        self.parse_select(subselect['subquery']['SelectStmt'])
        # TODO: deal with the targets
        self.table_info[alias_name] = Table(table_name, alias_name, False)

    def parse_from_clause(self, from_clause):
        for i in range(len(from_clause)):
            _table = from_clause[i]
            if 'RangeVar' in _table.keys():
                range_var = _table['RangeVar']
                self.parse_range_var(range_var)
            elif 'RangeSubselect' in _table.keys():
                subselect = _table['RangeSubselect']
                self.parse_range_subselect(subselect)

    def is_original_column(self, t_n, col_name):
        table_name = t_n
        if table_name == "":
            for t in self.table_info.keys():
                columns = self.table_info[t].columns.keys()
                if col_name in self.table_info[t].columns.keys():
                    return t, self.table_info[t].columns[col_name].is_origin
        else:
            col_info = self.table_info[t_n].columns[col_name]
            return table_name, col_info.is_origin
        return "", False

    def parse_lr_expr(self, expr, is_or, is_target):
        etype, table_name1, col_name1, value_type1, value1, is_o = 10, "", "", "", "", True
        if type(expr).__name__ == 'list':
            return etype, table_name1, col_name1, value_type1, value1, is_o
        if 'ColumnRef' in expr.keys():
            etype = 0
            column_ref = expr['ColumnRef']
            if len(column_ref['fields']) == 1:
                col_name1 = column_ref['fields'][0]['String']['str']
                table_name1, is_o = self.is_original_column("", col_name1)
                self.table_info[table_name1].add_used_column(col_name1)
            else:
                col_name1 = column_ref['fields'][1]['String']['str']
                table_name1, is_o = self.is_original_column(column_ref['fields'][0]['String']['str'],
                                                            col_name1)
                self.table_info[table_name1].add_used_column(col_name1)

        elif 'A_Const' in expr.keys():
            etype = 1
            const_info = expr['A_Const']
            val = const_info['val']
            if 'Integer' in val.keys():
                value_type1 = "int"
                value1 = str(val['Integer']['ival'])
            elif 'String' in val.keys():
                value_type1 = "str"
                value1 = val['String']['str']
            elif 'Float' in val.keys():
                value_type1 = "float"
                value1 = val['Float']['str']
        elif 'TypeCast' in expr.keys():
            etype = 1
            detial = expr['TypeCast']
            type_name = detial['typeName']
            value_type1 = type_name['TypeName']['names'][0]['String']['str']
            value1 = detial['arg']['A_Const']['val']['String']['str']
        elif 'SubLink' in expr.keys():
            etype = 1
            is_o = True
            self.parse_select(expr['SubLink']['subselect']['SelectStmt'])
        elif 'A_Expr' in expr.keys():
            etype = 4
            is_o = False
            is_c = self.parse_a_expr(expr['A_Expr'], is_or, is_target)
            if is_c:
                etype = 1
                is_o = True
        elif 'FuncCall' in expr.keys():
            self.parse_fun_call(expr['FuncCall'], is_or, is_target)

        return etype, table_name1, col_name1, value_type1, value1, is_o

    def parse_a_expr(self, a_expr, is_in_or, is_target):
        # currently, we only consider =/>/<
        op = a_expr['name'][0]['String']['str']
        # deal with left
        left_expr = a_expr['lexpr']
        etype1, table_name1, col_name1, value_type1, value1, is_o1 = self.parse_lr_expr(left_expr, is_in_or, is_target)
        right_expr = a_expr['rexpr']
        etype2, table_name2, col_name2, value_type2, value2, is_o2 = self.parse_lr_expr(right_expr, is_in_or, is_target)
        if etype1 == 1 and etype2 == 1:
            return True
        etype = etype1 + etype2
        # print(a_expr)
        if not is_target and op in self.support_op.keys() and is_o1 and is_o2 and (etype == 1 or etype == 0) and not is_in_or:
            if etype == 0:
                # table_name, _type, value_type, otherside):
                c_l = Condition(table_name1, "join", col_name1, "", table_name2 + "#@#" + col_name2)
                c_r = Condition(table_name2, "join", col_name2, "", table_name1 + "#@#" + col_name1)
                self.table_info[table_name1].join_conditions.append(c_l)
                self.table_info[table_name2].join_conditions.append(c_r)
            else:
                if etype1 == 0:
                    c = Condition(table_name1, op, col_name1, value_type2, value2)
                    self.table_info[table_name1].conditions.append(c)
                else:
                    c = Condition(table_name2, op, col_name2, value_type1, value1)
                    self.table_info[table_name2].conditions.append(c)

    def parse_bool_expr(self, bool_expr, is_in_or):
        is_or = is_in_or
        if bool_expr['boolop'] == 1:
            is_or = True
        args = bool_expr['args']
        for i in range(len(args)):
            arg = args[i]
            if 'A_Expr' in arg.keys():
                self.parse_a_expr(arg['A_Expr'], is_or, False)
            if 'BoolExpr' in arg.keys():
                self.parse_bool_expr(arg['BoolExpr'], is_or)
            if 'SubLink' in arg.keys():
                self.parse_select(arg['SubLink']['subselect']['SelectStmt'])

    def parse_where_clause(self, where_clause):
        # op 0:and 1:or, do not consider 'or'
        if 'BoolExpr' in where_clause.keys():
            bool_expr = where_clause['BoolExpr']
            self.parse_bool_expr(bool_expr, False)
        elif 'A_Expr' in where_clause.keys():
            a_expr = where_clause['A_Expr']
            self.parse_a_expr(a_expr, False, False)

    def parse_fun_call(self, fun_call, is_in_or, is_target):
        if 'agg_star' in fun_call.keys():
            return
        args = fun_call['args']
        for i in range(len(args)):
            if 'ColumnRef' in args[i].keys():
                self.parse_column_in_target(args[i]['ColumnRef'])
            elif 'A_Expr' in args[i].keys():
                self.parse_a_expr(args[i]['A_Expr'], is_in_or, is_target)

    def parse_column_in_target(self, column_info):
        # print(column_info)
        if len(column_info['fields']) == 1:
            if 'A_Star' in column_info['fields'][0].keys():
                return
            col_name = column_info['fields'][0]['String']['str']
            table_name, is_or = self.is_original_column('', col_name)
            if table_name == "":
                return
            self.table_info[table_name].add_used_column(col_name)
        else:
            table_name = column_info['fields'][0]['String']['str']
            col_name = column_info['fields'][1]['String']['str']
            self.table_info[table_name].add_used_column(col_name)

    def parse_res_target(self, res_target):
        # name in target, we do not need to consider. because it is for users.
        if 'FuncCall' in res_target['val'].keys():
            fun_call = res_target['val']['FuncCall']
            self.parse_fun_call(fun_call, False, True)
        elif 'ColumnRef' in res_target['val'].keys():
            column_info = res_target['val']['ColumnRef']
            self.parse_column_in_target(column_info)
        elif 'A_Expr' in res_target['val'].keys():
            self.parse_a_expr(res_target['val']['A_Expr'], False, True)

    def parse_targets(self, target_list):
        for i in range(len(target_list)):
            target = target_list[i]
            if 'ResTarget' in target.keys():
                res_target = target['ResTarget']
                self.parse_res_target(res_target)

    def parse_group_clause(self, group_by):
        which_table = ""
        columns = dict()
        index = 0
        for i in range(len(group_by)):
            if 'ColumnRef' in group_by[i].keys():
                column_ref = group_by[i]['ColumnRef']
                if i == 0:
                    if len(column_ref['fields']) == 1:
                        col_name1 = column_ref['fields'][0]['String']['str']
                        table_name1, is_o = self.is_original_column("", col_name1)
                        if table_name1 == "":
                            return
                        self.table_info[table_name1].add_used_column(col_name1)
                        columns[col_name1] = index
                        index += 1
                    else:
                        col_name1 = column_ref['fields'][1]['String']['str']
                        table_name1 = column_ref['fields'][0]['String']['str']
                        if not self.table_info[table_name1].is_origin:
                            return
                        self.table_info[table_name1].add_used_column(col_name1)
                        columns[col_name1] = index
                        index += 1
                    which_table = table_name1
                else:
                    if len(column_ref['fields']) == 1:
                        col_name1 = column_ref['fields'][0]['String']['str']
                        table_name1, is_o = self.is_original_column("", col_name1)
                        if table_name1 == "":
                            return
                        if table_name1 != which_table:
                            return
                        self.table_info[table_name1].add_used_column(col_name1)
                        columns[col_name1] = index
                        index += 1
                    else:
                        col_name1 = column_ref['fields'][1]['String']['str']
                        table_name1 = column_ref['fields'][0]['String']['str']
                        if table_name1 != which_table:
                            return
                        if not self.table_info[table_name1].is_origin:
                            return
                        self.table_info[table_name1].add_used_column(col_name1)
                        columns[col_name1] = index
                        index += 1
            else:
                return
        self.table_info[which_table].group = columns

    def parse_sort_clause(self, sort):
        which_table = ""
        columns = dict()
        index = 0
        for i in range(len(sort)):
            if 'SortBy' in sort[i].keys():
                node = sort[i]['SortBy']['node']
                if 'ColumnRef' in node.keys():
                    column_ref = node['ColumnRef']
                    if i == 0:
                        if len(column_ref['fields']) == 1:
                            col_name1 = column_ref['fields'][0]['String']['str']
                            table_name1, is_o = self.is_original_column("", col_name1)
                            if table_name1 == "":
                                return
                            self.table_info[table_name1].add_used_column(col_name1)
                            columns[col_name1] = index
                            index += 1
                        else:
                            col_name1 = column_ref['fields'][1]['String']['str']
                            table_name1 = column_ref['fields'][0]['String']['str']
                            if not self.table_info[table_name1].is_origin:
                                return
                            self.table_info[table_name1].add_used_column(col_name1)
                            columns[col_name1] = index
                            index += 1
                        which_table = table_name1
                    else:
                        if len(column_ref['fields']) == 1:
                            col_name1 = column_ref['fields'][0]['String']['str']
                            table_name1, is_o = self.is_original_column("", col_name1)
                            if table_name1 == "":
                                return
                            if table_name1 != which_table:
                                return
                            self.table_info[table_name1].add_used_column(col_name1)
                            columns[col_name1] = index
                            index += 1
                        else:
                            col_name1 = column_ref['fields'][1]['String']['str']
                            table_name1 = column_ref['fields'][0]['String']['str']
                            if table_name1 != which_table:
                                return
                            if not self.table_info[table_name1].is_origin:
                                return
                            self.table_info[table_name1].add_used_column(col_name1)
                            columns[col_name1] = index
                            index += 1
        self.table_info[which_table].order = columns

    def parse_select(self, select_stmt):
        # (1)parse from
        from_clause = select_stmt['fromClause']
        self.parse_from_clause(from_clause)

        # (2)parse where
        if 'whereClause' in select_stmt.keys():
            where_clause = select_stmt['whereClause']
            self.parse_where_clause(where_clause)

        # (3)parse target
        target_list = select_stmt['targetList']
        self.parse_targets(target_list)

        # (4)parse order,group
        if 'groupClause' in select_stmt.keys():
            self.parse_group_clause(select_stmt['groupClause'])

        if 'sortClause' in select_stmt.keys():
            self.parse_sort_clause(select_stmt['sortClause'])

