# HyperLogLog implementation in Python

This Python module implements the HyperLogLog algorithm using 32bit and 64bit hash functions.

It is based on the following papers:
- HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm - P. Flajolet, E. Fusy, O. Gandouet, & 
 F. Meunier (http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)
- HyperLogLog in Practice: Algorithmic Engineering of a State of The Art Cardinality Estimation Algorithm - S. Heule,
 M. Nunkesser, A. Hall (https://stefanheule.com/papers/edbt13-hyperloglog.pdf)
- Appendix to HyperLogLog in Practice: Algorithmic Engineering of a State of the Art Cardinality Estimation Algorithm -
 S. Heule, M. Nunkesser, A. Hall (http://goo.gl/iU8Ig)

 Mentionned papers are enclosed in /References

==========================================================================
1. Examples to use this HLL implementation are in the file 'examples.py'. The functions are implemented using the other scripts from /HLL.

2. Unit tests are implemented in /tests using Pytest. They can be launched automatically from run_tests.sh.
This requires installing Pytest ($pip install pytest). To use pyspark, we also use findspark ($pip install findspark)

==========================================================================
The papers mentions a further improvement of HyperLogLog algorithm, based on the observation that the hashmap is sparse when estimating small cardinalities. For low orders, we then replace the hashmap as a list by a key-pair mapping. This improvement aims at reducing the memory usage. In addition to the sparse hmap, It requires to go down at the bit level to use only the minimum require number of bits (otherwise we have no control over the memory usage). However,  Python is a high level language that doesn't allow such low level memory management.
