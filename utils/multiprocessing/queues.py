#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from multiprocessing import current_process
from Queue import Empty
from time import time
import zmq
import numpy as np


class ZMQQueue(object):
    """ZMQ-based Queue
        A queue used to trans data between process
    """
    def __init__(self, buffsize, mode=None, connect_mode='one2one'):
        """
        Parameters
        ----------
        buffsize : int
            This is the same as the buffsize in  multiprocessing.Queue.
        mode : str or None
            If mode is 'ndarray', the message is passed without pickling. Otherwise
            passes the message as a default python object.
        connect_mode : str
            One of 'one2one', 'one2multi' and 'multi2one'. This defines the numbers
            of producers and consumers.
        """
        self._buffsize = buffsize
        self._mode = mode
        self._connect_mode = connect_mode
        if self._connect_mode not in ('one2one', 'one2multi', 'multi2one'):
            raise ValueError("Unsupported connect mode")

        self._cpid = current_process().pid
        queue_num = int(int(time() * 1e6) % 1e12)
        self._path = 'ipc:///tmp/parrots_queue_{}_{}'.format(self._cpid,
                                                             queue_num)
        self._socket = None
        self._pid = None
        self._for_put = None

    def _check_setup(self):
        if current_process().pid != self._pid:
            self._pid = None
            self._for_put = None

    def _setup_get(self):
        if self._pid is not None:
            raise RuntimeError('cannot setup twise')
        self._pid = current_process().pid
        self._for_put = False

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PULL)
        self._socket.setsockopt(zmq.RCVHWM, self._buffsize)
        if self._connect_mode in ('one2one', 'one2multi'):
            self._socket.connect(self._path)
        else:
            self._socket.bind(self._path)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

    def _setup_put(self):
        if self._pid is not None:
            raise RuntimeError('cannot setup twise')
        self._pid = current_process().pid
        self._for_put = True

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        if self._connect_mode in ('one2one', 'one2multi'):
            self._socket.bind(self._path)
        else:
            self._socket.connect(self._path)

    def _recv(self):
        if self._mode != 'ndarray':
            data = self._socket.recv_pyobj()
            end = self._socket.recv_pyobj()
            assert end == 'end'
            return data

        assert self._mode == 'ndarray'
        md = self._socket.recv_json()
        idx = md['idx']
        data_type = md['type']
        metas = md['metas']
        result = []
        if data_type == 'dict':
            result = {}
        for meta in metas:
            msg = self._socket.recv()
            buf = buffer(msg)
            a = np.frombuffer(buf, dtype=meta['dtype'])
            a = a.reshape(meta['shape'])
            if meta['t']:
                a = a.T
            if data_type == 'dict':
                key = str(meta['key'])
                assert key is not None
                result[key] = a
            else:
                result.append(a)
        end = self._socket.recv_json()
        assert end == 'end'
        if data_type == 'ndarray':
            assert isinstance(result, list) and len(result) == 1
            result = result[0]
            assert isinstance(result, np.ndarray)
        elif data_type == 'tuple':
            result = tuple(result)
        return idx, result

    def _send(self, data):
        if self._mode != 'ndarray':
            self._socket.send_pyobj(data, zmq.SNDMORE)
            return self._socket.send_pyobj('end')
   
        assert self._mode == 'ndarray'
        idx, result = data
        assert isinstance(result, (dict, list, np.ndarray, tuple))
        if isinstance(result, dict):
            data_type = 'dict'
        elif isinstance(result, list):
            data_type = 'list'
        elif isinstance(result, np.ndarray):
            data_type = 'ndarray'
        else:
            assert isinstance(result, tuple)
            data_type = 'tuple'

        metas = []
        to_send = []
        if data_type == 'ndarray':
           result = [result,]
        for array in result:
            key = None
            if data_type == 'dict':
                assert isinstance(array, str)
                key = array
                array = result[key]
            assert isinstance(array, np.ndarray)
            if array.flags['C']:
                t = False
            elif array.flags['F']:
                t = True
                array = array.T
            else:
                t = False
                array = np.ascontiguousarray(array)
            assert array.flags['C']
            metas.append(dict(
                    t=t,
                    dtype=str(array.dtype),
                    shape=array.shape,
                    key = key
            ))
            to_send.append(array)
        md = dict(
            idx=idx,
            type=data_type,
            metas=metas
        )
        self._socket.send_json(md, zmq.SNDMORE)
        for a in to_send:
            assert isinstance(a, np.ndarray)
            self._socket.send(a, zmq.SNDMORE, copy=False)
        return self._socket.send_json('end')

    def get(self, timeout=None):
        self._check_setup()
        if self._pid is None:
            self._setup_get()
        if self._for_put:
            raise RuntimeError('cannot call get from put end')
        if timeout is not None:
            timeout = timeout * 1000

        # get
        socks = dict(self._poller.poll(timeout))
        if socks.get(self._socket) == zmq.POLLIN:
            return self._recv()
        else:
            raise Empty

    def put(self, data):
        self._check_setup()
        if self._pid is None:
            self._setup_put()
        if not self._for_put:
            raise RuntimeError('cannot call put from get end')

        # put
        return self._send(data)

    def close(self):
        self._socket.close()

    def join_thread(self):
        pass
