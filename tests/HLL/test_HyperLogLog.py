# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 23:04:22 2016
@author: pat

Tests of the HLL.HyperLogLog module.
"""

#==============================================================================
# Imports
#==============================================================================
import os
import json
import pytest

from HLL.HyperLogLog import *


#==============================================================================
# Utilities and Global definitions
#==============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATAFILE = os.path.join(CURRENT_DIR, 'data.txt')


#Normally, the following function would be in a separate utility module, and
#this module would be tested itself. For simplicity, I kept them here in our
#particular case.
def non_zero_idx_val(seq):
    """
    Return a list of tuple (index, value) for non-zero element of seq
    
    seq - a sequence of numerical value
    The expectation is that seq is sparse
    """
    return [(i, v) for i, v in enumerate(seq) if v > 0]


#==============================================================================
# Classes
#==============================================================================

class Test_HyperLogLog(object):
    """
    Unit tests for the HyperLogLog and the HyperLogLog64 classes
    """
    
    def setup(self):
        self.data = ['abc', 'def', 'ghi']
        self.data_duplicate = self.data + ['def', 'abc']
        self.num_data = [11, 11.717, 12.207, 12.7896, 11]
        self.colliding_data = [654941.845, 400.2024]
        self.reset_hll()
        
    def tearDown(self):
        self.reset_hll()
        
    def reset_hll(self):
        """
        Resst the hlls instance so that we can continue testing
        """
        self.hll = HyperLogLog(250)
        self.hll64 = HyperLogLog64(2**17)
    
    def test_init_error_handling(self):
        """
        Tests error handling when initiating a HLL
        """
        with pytest.raises(ValueError) as err:
            hll = HyperLogLog(2)
        assert err.value.message == "k=2 should be in range [16, 65536]"
        with pytest.raises(ValueError) as err:
            hll = HyperLogLog(2**17)
        assert err.value.message == "k=131072 should be in range [16, 65536]"
        hll = HyperLogLog(2**16)
        assert hll.k == 2**16
        hll = HyperLogLog64(2**17)
        assert hll.k == 2**17
        
    def test_parameters(self):
        """
        Tests HLL parameters initialization
        """
        assert self.hll.p == 8
        assert self.hll.m == 256
        assert round(self.hll.alpha - 0.7182725932495458, 5) == 0
        assert round(self.hll.error - 0.065, 5) == 0
        assert self.hll64.treshold == 120000
    
    def test_hmaps(self):
        """
        Tests the hmaps population functionnalities
        """
        #Single element insertion
        self.hll.append(self.data[0])
        assert non_zero_idx_val(hll.hmap) == [(54, 1)]
        #Multiple distinct element insertions
        self.hll.extend(self.data)
        assert non_zero_idx_val(hll.hmap) == [(51, 2), (54, 1), (214, 2)]
        self.reset_hll()
        #Element insertions with duplicates
        self.hll.extend(self.data_duplicate)
        assert non_zero_idx_val(hll.hmap) == [(51, 2), (54, 1), (214, 2)]
        self.reset_hll()
        #Element insertions with numerical values
        self.hll.extend(self.num_data)
        assert non_zero_idx_val(hll.hmap) == [(17, 3), (144, 2), (145, 4),
                                              (182, 2)]
        self.reset_hll()
        #Test the key collision handling (keep max value)
        self.hll.append(self.colliding_data[0])
        assert non_zero_idx_val(hll.hmap) == [(0, 1)]        
        self.hll.append(self.colliding_data[1])
        assert non_zero_idx_val(hll.hmap) == [(0, 2)]
        self.reset_hll()
        self.hll.append(self.colliding_data[1])
        assert non_zero_idx_val(hll.hmap) == [(0, 2)]        
        self.hll.append(self.colliding_data[0])
        assert non_zero_idx_val(hll.hmap) == [(0, 2)]
        self.reset_hll()
    
    def test_hmaps(self):
        """
        Tests the hmaps population functionnalities
        """
        #Merging and adding, with commutativity, without collision
        hll1 = HyperLogLog(250)
        hll1.extend(self.data)
        hll2 = HyperLogLog(250)
        hll2.extend(self.num_data)
        test_set = set(non_zero_idx_val(hll1.hmap)).union(
                                            set(non_zero_idx_val(hll2.hmap)))
        hll1_prime = HyperLogLog(250) #merging
        hll1_prime.extend(self.data)
        hll1_prime.merge(hll2)
        assert set(non_zero_idx_val(hll1_prime.hmap)) == test_set
        hll2_prime = HyperLogLog(250) #merging commutativity
        hll2_prime.extend(self.num_data)
        hll2_prime.merge(hll1)
        assert set(non_zero_idx_val(hll2_prime.hmap)) == test_set
        hll3 = hll1 + hll2 #addition
        assert set(non_zero_idx_val(hll3.hmap)) == test_set
        hll4 = hll2 + hll1 #addition commutativity
        assert set(non_zero_idx_val(hll4.hmap)) == test_set
        
        #Collision testing
        hll1 = HyperLogLog(250)
        hll1.append(self.colliding_data[0])
        hll2 = HyperLogLog(250)
        hll2.append(self.colliding_data[1])
        hll1_prime = HyperLogLog(250)
        hll1_prime.append(self.colliding_data[0])
        hll1_prime.merge(hll2)
        assert hll1_prime.hmap[0] == 2
        hll2_prime = HyperLogLog(250)
        hll2_prime.append(self.colliding_data[1])
        hll2_prime.merge(hll2)
        assert hll2_prime.hmap[0] == 2
        assert (hll1 + hll2).hmap[0] == 2
        assert (hll2 + hll1).hmap[0] == 2
        
    def test_cardinality(self):
        """
        Tests the cardinality estimation
        """
        #At low order
        self.hll.extend(self.data)
        assert round(self.hll.cardinality - 3.017716672522796, 5) == 0
        self.hll64.extend(self.data)
        assert round(self.hll64.cardinality - 3.0000343327992325, 5) == 0
        self.reset_hll()
        #At high order
        with open(DATAFILE, 'r') as f:
            data = json.load(f)
        return data
        self.hll.extend(data)
        assert round(self.hll.cardinality - 695.1859783711712, 5) == 0
        self.hll64.extend(data)
        assert round(self.hll64.cardinality - 638.5529193179921, 5) == 0
        self.reset_hll()

       