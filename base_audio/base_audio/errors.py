# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:01:35 2020

@author: amarmore

Module defining specific errors to be raised
"""

class ArgumentException(BaseException): pass
class InvalidArgumentValueException(ArgumentException): pass

class AbnormalBehaviorException(BaseException): pass
class ToDebugException(AbnormalBehaviorException): pass
class BuggedFunctionException(AbnormalBehaviorException): pass
class OutdatedBehaviorException(AbnormalBehaviorException): pass


