import numpy as np
from Utility import PostgreSQL as pg
import math
from typing import List
import sys

max_index_size = 4


class Env:
    def __init__(self, workload, candidates, mode, a):
        self.workload = workload
        self.candidates = candidates
        # create real/hypothetical index
        self.mode = mode
        self.pg_client1 = pg.PGHypo()
        self.pg_client2 = pg.PGHypo()
        self._frequencies = [1265, 897, 643, 1190, 521, 1688, 778, 1999, 1690,1433, 1796, 1266, 1046, 1353]
        self.frequencies = np.array(self._frequencies) / np.array(self._frequencies).sum()

        # state info
        self.init_cost = np.array(self.pg_client1.get_queries_cost(workload))*self.frequencies
        self.init_cost_sum = self.init_cost.sum()
        # self.init_state = np.append(self.init_cost, np.zeros((len(candidates),), dtype=np.float))
        self.init_state = np.append(self.frequencies, np.zeros((len(candidates),), dtype=np.float))
        self.last_state = self.init_state
        self.last_cost = self.init_cost
        self.last_cost_sum = self.init_cost_sum

        # utility info
        self.index_oids = np.zeros((len(candidates),), dtype=np.int)
        self.performance_gain = np.zeros((len(candidates),), dtype=np.float)
        self.current_index_count = 0
        self.currenct_index = np.zeros((len(candidates),), dtype=np.float)
        self.current_index_storage = np.zeros((len(candidates),), dtype=np.float)

        # monitor info
        self.cost_trace_overall = list()
        self.index_trace_overall = list()
        self.min_cost_overall = list()
        self.min_indexes_overall = list()
        self.current_min_cost = (np.array(self.pg_client1.get_queries_cost(workload))*0.1*self.frequencies).sum()
        self.current_min_index = np.zeros((len(candidates),), dtype=np.float)

        self.current_storage_sum = 0
        self.max_count = 0
        self.a = a

        self.pre_create = []

    def checkout(self):
        pre_is = []
        while True:
            current_max = 0
            current_index = None
            current_index_len = 0
            start_sum = (np.array(self.pg_client2.get_queries_cost(self.workload)) * self.frequencies).sum()
            for index in self.candidates:
                oid = self.pg_client2.execute_create_hypo(index)
                cu_sum = (np.array(self.pg_client2.get_queries_cost(self.workload)) * self.frequencies).sum()
                x = (start_sum - cu_sum)/start_sum
                if x > 0.4 and current_max < x:
                    current_max = x
                    current_index = index
                    current_index_len = current_index_len
                self.pg_client2.execute_delete_hypo(oid)
            if current_index is None:
                break
            pre_is.append(current_index)
            self.pg_client2.execute_create_hypo(current_index)
        # pre_is = ['lineitem#l_orderkey,l_shipdate', 'lineitem#l_partkey,l_orderkey', 'lineitem#l_receiptdate', 'lineitem#l_shipdate,l_partkey', 'lineitem#l_suppkey,l_commitdate']
        # pre_is = ['lineitem#l_orderkey,l_suppkey', 'lineitem#l_partkey,l_suppkey', 'lineitem#l_receiptdate', 'lineitem#l_shipdate,l_discount', 'lineitem#l_suppkey,l_commitdate']
        # pre_is.append('lineitem#l_orderkey')
        self.pre_create = pre_is
        self.pg_client2.delete_indexes()
        self.max_count -= len(self.pre_create)
        return pre_is

    def step(self, action):
        action = action[0]
        if self.currenct_index[action] != 0.0:
            # self.cost_trace_overall.append(self.last_cost_sum)
            # self.index_trace_overall.append(self.currenct_index)
            return self.last_state, 0, False

        self.index_oids[action] = self.pg_client1.execute_create_hypo(self.candidates[action])
        self.currenct_index[action] = 1.0
        oids : List[float] = list()
        oids.append(self.index_oids[action])
        storage_cost = self.pg_client1.get_storage_cost(oids)[0]
        # print(storage_cost)
        self.current_storage_sum += storage_cost
        self.current_index_storage[action] = storage_cost
        self.current_index_count += 1

        # reward & performance gain
        current_cost_info = np.array(self.pg_client1.get_queries_cost(self.workload))*self.frequencies
        current_cost_sum = current_cost_info.sum()
        # performance_gain_current = self.init_cost_sum - current_cost_sum
        # performance_gain_current = (self.last_cost_sum - current_cost_sum)/self.last_cost_sum
        # performance_gain_avg = performance_gain_current.round(1)
        # self.performance_gain[action] = performance_gain_avg
        # monitor info
        # self.cost_trace_overall.append(current_cost_sum)

        # update
        self.last_cost = current_cost_info
        # state = (self.init_cost - current_cost_info)/self.init_cost
        self.last_state = np.append(self.frequencies, self.currenct_index)
        deltac0 = (self.init_cost_sum - current_cost_sum) / self.init_cost_sum
        deltac1 = (self.last_cost_sum - current_cost_sum)/self.init_cost_sum
        # print(deltac0)
        '''deltac0 = max(0.000003, deltac0)
        if deltac0 == 0.000003:
            reward = -10
        else:
            reward = math.log(0.0003, deltac0)'''
        b = 1 - self.a
        #reward = deltac0
        print(deltac0)
        reward =self.a*deltac0*100 + b*deltac1*100
        # reward = deltac0
        # reward = math.log(0.99, deltac0)
        '''deltac0 = self.init_cost_sum/current_cost_sum
        deltac1 = self.last_cost_sum/current_cost_sum
        reward = math.log(deltac0,10)'''
        self.last_cost_sum = current_cost_sum
        if self.current_index_count >= self.max_count:
            self.cost_trace_overall.append(current_cost_sum)
            self.index_trace_overall.append(self.currenct_index)
            return self.last_state, reward, True
        else:
            return self.last_state, reward, False
            # re5 return self.last_state, reward, False

    def reset(self):
        self.last_state = self.init_state
        self.last_cost = self.init_cost
        self.last_cost_sum = self.init_cost_sum
        # self.index_trace_overall.append(self.currenct_index)
        self.index_oids = np.zeros((len(self.candidates),), dtype=np.int)
        self.performance_gain = np.zeros((len(self.candidates),), dtype=np.float)
        self.current_index_count = 0
        self.current_min_cost = np.array(self.pg_client1.get_queries_cost(self.workload)).sum()
        self.current_min_index = np.zeros((len(self.candidates),), dtype=np.float)
        self.currenct_index = np.zeros((len(self.candidates),), dtype=np.float)
        self.current_index_storage = np.zeros((len(self.candidates),), dtype=np.float)
        self.pg_client1.delete_indexes()
        # self.cost_trace_overall.append(self.last_cost_sum)
        if len(self.pre_create) > 0:
            print("x")
            for i in self.pre_create:
                self.pg_client1.execute_create_hypo(i)
            self.init_cost_sum = (np.array(self.pg_client1.get_queries_cost(self.workload))*self.frequencies).sum()
            self.last_cost_sum = self.init_cost_sum
        return self.last_state