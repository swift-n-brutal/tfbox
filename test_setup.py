import tensorflow as tf
import numpy as np

def add_bias(x, bias=1.):
    b = tf.get_variable('bias', shape=(1,), initializer=tf.constant_initializer(bias))
    o = x + b
    return o, b

class Model(object):
    def setup(self, name='M'):
        self.name = name
        self.params = list()
        with tf.variable_scope(self.name) as scope:
            self.scope = scope
            inp = tf.placeholder(tf.float32, name='input', shape=(1,))
        self.input = inp
        self.output = self.run(inp, True)

    def run(self, inp, init=False):
        with tf.variable_scope(self.scope, reuse=not init) as scope:
#            if not init:
#                scope.reuse_variables()
            o, b = add_bias(inp)
            if init:
                self.params.append(b)
        return o

def main():
    md = Model()
    md.setup()
    out = md.run(10.)
    ss = tf.Session()
    ss.run(tf.global_variables_initializer())
    print ss.run(out)
    print ss.run(md.output, feed_dict={md.input: [100.0]})
    update = md.params[0].assign(md.params[0] + 20.0)
    ss.run(update)
    out3 = md.run(1000)
    print ss.run(out3)
    print ss.run(md.output, feed_dict={md.input: [10000.0]})

    print md.input.name, md.output.name
    print out.name, out3.name
    print md.params[0].name

if __name__ == '__main__':
    main()
