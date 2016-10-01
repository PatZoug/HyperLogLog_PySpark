# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 18:28:38 2016
@author: pat
"""

#==============================================================================
# Imports
#==============================================================================
import HyperLogLog as HLL


#==============================================================================
# Cardinality estimation
#==============================================================================
def estimate_distinct_elements(items, k):
    """
    Returns an estimate to the number of distinct elements in items
    
    items - a sequence of elements
    k - number of hash functions
    """
    hll = HLL.HyperLogLog64(k)
    hll.extend(items)
    return hll.cardinality
    

#==============================================================================
# Parallelisation
#==============================================================================
#In the following, we assume estimate_distinct_elements_parallel is to be used
#in PySpark, with a SparkContext previously defined.
#This is typically what we obtain when launching PySpark from the shell:
#$./bin/pyspark --py-files PATH/TO/ZIPPED/HLL/MODULE.zip
#Followed by, in the Python console:
#from examples import *


def init_compute_hmaps(k):
    """
    Return a method yielding hmaps from hll initialised with k hash functions
    """
    def compute_hmaps(list_of_sequences):
        """
        Iterator yielding 1 HyperLogLog.hmap per sequence in given iterable
        
        list_of_sequences - iterable of iterable
        """
        for sequence in list_of_sequences:
            hll = HLL.HyperLogLog64(k)
            hll.extend(sequence)
            yield hll
    return compute_hmaps


def estimate_distinct_elements_parallel(lists_of_items, k, spark_context):
    """
    Returns an estimate to the number of distinct elements in items

    items - a sequence of elements
    k - number of hash functions
    spark_context - a spark context
    """
    hll = spark_context.parallelize(lists_of_items) \
                       .mapPartitions(init_compute_hmaps(k)) \
                       .reduce(lambda x, y :x + y)
    return hll.cardinality


def calculate_empirical_accuracy(items, estimate, spark_context, 
                                 relative=True):
    """
    Returns the difference between the estimate and the actual cadinality
    
    items - a sequence of elements
    estimate - a number that represents the estimate
    relative - a boolean: if true relative error, else absolute error
    spark_context - a spark context
    """
    card = spark_context.parallelize(items).distinct().count()
    if relative:
        return float(card - estimate) / float(card)
    return float(card - estimate)
