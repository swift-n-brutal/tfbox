# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:31:38 2017

@author: shiwu_001
"""

import signal
class Handler:
    def __init__(self, idx=-1):
        self.alive = True
        self.idx = idx
        signal.signal(signal.SIGINT, self.on_interruption)
        signal.signal(signal.SIGTERM, self.on_termination)

    def on_interruption(self, sig, frame):
        pass
        #print self.idx, 'Got SIGINT'

    def on_termination(self, sig, frame):
        #print self.idx, 'Got SIGTERM'
        self.alive = False
