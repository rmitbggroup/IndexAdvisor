# Index Advisor based on Deep Reinforcement Learning
Code for CIKM2020 [paper](https://dl.acm.org/doi/abs/10.1145/3340531.3412106)

# What does it do?
This is an index advisor tool to recommend an index configuration for a certain workload under maximum storage or index number. It combines the heuristic rules and deep reinforcement learning together.

# What do I need to run it?
1. You should install a [PostgreSQL](https://www.postgresql.org/) database instance with [HypoPG extension](https://hypopg.readthedocs.io/en/latest/).
2. You should install the required python packages (see environment.yaml exported from conda).
3. In this code, we adopt TPC-H. Thus, you construct your own TPC-H database instance. 
4. We need the TPC-H tool to generate the workload. You can download it from this [page](http://tpc.org/tpc_documents_current_versions/current_specifications5.asp).

# How do I run it?
1. You can find the entry in Entry/EntryM3DP.py
2. There is a sample about how to use the workload and index candidates generation algorithms in Utility/Sample4GenCandidates.py.

# Notice
1. The index candidates generated algorithms (parser and generation algorithms in Utility/ParserForIndex.py) are for TPC-H cases. It may be not suitable TPC-DS. Because some query patterns are not in TPC-H.
