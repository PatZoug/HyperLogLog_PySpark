# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 21:21:21 2016
@author: Pat

This module implements several version of the HyperLogLog algorithm, one with
a 32bit hash function and one with a 64bit hash function.
    
HyperLogLog algorithm is based on the observation that, given N distinct
elements and a "good" hashing function from the element space to [0, 1], the 
expectation e for the distance between 0 and the first element hashcode is:
e = 1 / (N + 1).
And then: N = (e - 1) / e
"""

#==============================================================================
# Imports
#==============================================================================
import sys
import hashlib
import math
from bisect import bisect

from HLL64Constants import rawEstimateData, biasData, treshold


#==============================================================================
# Python 3 compatibility
#==============================================================================

#This script was developed using Python 2.7. However, in Python 3, the type
#long no longer exists (sys.maxint=2**63 - 1 (or =2**31 - 1 on 32bit OS) have 
#been removed). Since we are going to use numbers longer than this, we need to
#use long in Python 2.7, and int in Python 3.
if sys.version_info.major > 2:
    long = int


#==============================================================================
# Classes
#==============================================================================

class HyperLogLog(object):
    """
    HyperLogLog implementation for cardinality estimation of multiset

    This implementation uses 32 bit-long hashes and uses between 4 and 16 bits
    as keys to create a hashmap. The hashmap then contains between 2**4 and 
    2**16 key-value pairs. The user can choose the number of hashing functions
    he want to use within this range.
    """
    
    hlength = 32
    hex_length = 8 #hexadecimal length
    pmin = 4
    pmax = 16
    
    def __init__(self, k, hash_func=hashlib.sha1):
        """
        HyperLogLog initialization

        k - the size of the hashmap - we should have 2**4 <= k <= 2**16
        hash_func - the base hash function to use

        Note that the hash function should return at least a class.hlength bit 
        long hash code. Using hashlib.sha1 is recommended for the following 
        reasons:
        - it is a 160 bits long hash (more than sufficient in our case)
        - it outperforms sha224, sha256, sha512, etc (eg: 
        http://atodorov.org/blog/2013/02/05/performance-test-md5-sha1-sha256-sha512/)
        - md5 is known to present a lot of collisions
        """
        if int(k) < 2 ** self.pmin or int(k) > 2 ** self.pmax:
            msg = "k={0} should be in range [{1}, {2}]"
            raise ValueError(msg.format(k, 2 ** self.pmin, 2 ** self.pmax))
        self.k = int(k)
        self.hash_func = hash_func
        #We will use a p + 32 bits long hash. We will use the first p bits of
        #our hash as key for the hashmap. We will make sure to keep 32 bits
        #after that for leading zeros calculation.
        self.p = int(math.ceil(math.log(self.k, 2)))
        self.effective_k = 2 ** self.p
        self.m = int(2**self.p)
        self.hmap = [0] * self.m
        self.alpha = self.get_alpha()
        self.error = 1.04 / math.sqrt(2 ** self.p)
        
    def get_alpha(self):
        """
        Returns the alpha to compute the cardinality estimate (depending on k)
        
        a16 = 0.673, a32 = 0.697, a64 = 0.709,
        am = 0.7213 / (1 + 1.079/m) for m >=128
        """
        if self.m == 16:
            return 0.673
        if self.m == 32:
            return 0.697
        if self.m == 64:
            return 0.709
        return 0.7213 / (1 + 1.079 / self.m)

    def append(self, element):
        """
        Appends a new element to the HyperLogLog
        
        element - an element of the set to be estimated
        """
        #Encode the element to bits + we only take the first X bits needed
        val = self.hash_func(str(element).encode())
        val = long(val.hexdigest()[:self.hex_length], 16)
        if (self.p % 4) != 0:        
            val = val >> (4 - self.p % 4)
        #Use the self.p first bits as key to the hashmap
        hkey = val & (self.m - 1)
        #Use the remaining 32bit for the leading zero calculation
        hval = val >> self.p
        self.hmap[hkey] = max(self.hmap[hkey],
                              self.hlength - self.p - hval.bit_length() + 1)
    
    def extend(self, elements):
        """
        Appends several elements to the HyperLogLog

        elements - an iterable of elements of the set to be estimated
        
        This is only for commodity, it is not a bulk append
        """
        for element in elements:
            self.append(element)
    
    @property
    def _raw_estimate(self):
        """
        The estimate prior to any low order or high order corrections
        """
        return self.alpha * (self.m ** 2) / sum(2 ** (-i) for i in self.hmap)
    
    @property
    def cardinality(self):
        """
        The cardinality estimate with low and high order correction
        """
        E = self._raw_estimate
        # Low order correction
        if E <= 5 * self.m / 2:
            V = self.hmap.count(0) 
            if V > 0:
                #Linear counting estimate
                return self.m * math.log(self.m / float(V))
        #High order correction
        if E > (2 ** 32) / float(30):
            return - (2 ** 32) * log(1 - E / (2 ** 32))
        #Reguler estimate
        return E
    
    def merge(self, *others):
        """
        Merge several instances by merging their hashmaps
        
        others - a sequence of this class instance

        Useful to get the cardinality of merged multiset
        """
        for other in others:
            if self.k != other.k:
                raise Exception("Number k of hash functions must be equal.")
        self.hmap = [max(el) for el in zip(*[hll.hmap
                                            for hll in list(others) + [self]])]
    
    def __add__(self, other):
        """
        + operator

        other - an instance of this class
        """
        new_HLL = self.__class__(self.k)
        new_HLL.merge(self, other)
        return new_HLL

        
class HyperLogLog64(HyperLogLog):
    """
    HyperLogLog implementation using 64bit hash functions
    
    Since in classical HyperLogLog we use 32 hash functions, we got high order
    error due to hash collisions starting at around 2**32 (around 1 billions)
    elements. By using 64bit hash function, we remove this issue since it's 
    currently unlikely to hit 2**64 distincts elements. However bias at lower
    order increase and has to be corrected.
    Size of hashmap can be increase to 2**18.
    """
    
    hlength = 64
    hex_length = 16 #hexadecimal length
    pmax = 18
    _rawEstimateData = rawEstimateData
    _biasData = biasData
    _treshold = treshold
    
    def estimate_bias(self, E, k=6):
        """
        Estimated bias, depending on raw estimate and self.p

        E - the raw estimate of the cardinality
        k - the number of nearest neighbors to use to interpolate the bias
        
        The estimated bias is interpolated from biasData. The indexes to use in
        biasData are found from rawEstimateData.
        Heule and colleagues use k-nearest neighbors to interpolate, with k=6,
        but states that linear interpolation with the 2 nearest points "would
        give very similar results" (http://goo.gl/iU8Ig p.3). However, since 
        rawEstimateData sequences are not perfectly sorted
        (eg: rawEstimateData[1][127:130]), we prefere to implement the 
        6-nearest neighbors method to avoid any inconsistencies.
        """
        #First, let's observe that rawEstimatedData sequences are almost sorted.
        #We are then going to bisect the sequence, take the 2*k surrounding
        #idxs and return only the k closest ones.
        rED = self._rawEstimateData[self.p - 4]
        #bisect implements a binary search algorithm, which time complexity is
        #log(n + 1)
        idx = bisect(rED, E)
        #Thanks to Python flexible slicing (allowing to exceeed the 
        #[0, length] boundaries for the indexes), no need to handle edge cases.
        rED_slice = rED[idx - 2 * k:idx + 2 * k]
        dist = [((E - el)**2, i) for i, el in enumerate(rED_slice)]
        dist.sort()
        #Use the k nearest ones to interpolate the bias
        bias = self._biasData[self.p - 4]
        return sum(bias[i] for _, i in dist[:k]) / float(k)
    
    @property
    def treshold(self):
        """
        The treshold value depending on self.p
        """
        return self._treshold[self.p - 4]
    
    @property
    def cardinality(self):
        """
        The cardinality estimate with low and high order correction
        """
        V = self.hmap.count(0) 
        if V > 0:
            #Linear counting estimate
            H = self.m * math.log(self.m / float(V))    
            if H <= self.treshold:
                #return the Linear counting estimate if smaller than treshold
                return H
        E = self._raw_estimate
        if E <= 5 * self.m:
            E = E - self.estimate_bias(E)
        return E


