# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 23:04:22 2016

@author: pat
"""

#==============================================================================
# Imports
#==============================================================================
import findspark  # this needs to be the first import
findspark.init()

import os
import json
import pytest
import logging
from pyspark import SparkConf, SparkContext
#Please not that importing pyspark requires to have setup the SPARK_HOME 
#environment variable (probably in .bashrc if you are running Linux)
#Something like this: export SPARK_HOME="/home/pat/spark-1.6.1"

from HLL.examples import *

#==============================================================================
# Utilities and Global definitions
#==============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


#Normally, the following function would be in a separate utility module, and
#this module would be tested itself. For simplicity, I kept them here in our
#particular case.
def load_json(filepath):
    """
    Loads and returns the data from the specified json file
    
    filepath - the path to the json file
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


@pytest.fixture(scope="session")
def spark_context(request):
    """
    Fixture for creating a spark context
    
    request - pytest.FixtureRequest object
    """
    conf = (SparkConf().setMaster("local[2]").setAppName("pytest-pyspark-local-testing"))
    sc = SparkContext(conf=conf)
    request.addfinalizer(lambda: sc.stop())
    #Turn down spark logging for the test context
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)
    return sc

pytestmark = pytest.mark.usefixtures("spark_context")

#==============================================================================
# Classes
#==============================================================================

class Test_Examples(object):
    """
    Unit tests for the functions of the module examples.py
    """
    
    def setup(self):
        self.items = load_json(os.path.join(CURRENT_DIR, 'data.txt'))
        self.list_of_items = load_json(os.path.join(CURRENT_DIR, 'data2.txt'))
        self.k = 2**12
        
    def tearDown(self):
        self.items = []
        self.list_of_items = []
    
    def test_functions(self, spark_context):
        """
        Tests each functions
        """
        #estimate_distinct_elements
        assert round(estimate_distinct_elements(self.items, self.k) \
                     - 645.2579301443818, 5) == 0
        #init_compute_hmaps
        compute_hmaps = init_compute_hmaps(self.k)
        hlls = [hll for hll in compute_hmaps(self.list_of_items[:2])]
        assert len(hlls) == 2
        assert (hlls[0].k, hlls[1].k) == (self.k, self.k)
        hll0 = HLL.HyperLogLog64(self.k)
        hll0.extend(self.list_of_items[0])
        assert hll0.hmap == hlls[0].hmap
        hll1 = HLL.HyperLogLog64(self.k)
        hll1.extend(self.list_of_items[1])
        assert hll1.hmap == hlls[1].hmap
        #estimate_distinct_elements_parallel
        card = estimate_distinct_elements_parallel(self.list_of_items, self.k,
                                                   spark_context)
        assert round(card - 2875.28999, 5) == 0
        #calculate_empirical_accuracy
        card = estimate_distinct_elements(self.items, self.k)
        abs_err = calculate_empirical_accuracy(self.items, card,
                                               spark_context, relative=False)
        rel_err = calculate_empirical_accuracy(self.items, card,
                                               spark_context, relative=True)
        assert round(abs_err - (-7.257930144381817), 5) == 0
        assert round(rel_err - (-0.011376066057024792), 5) == 0
        
