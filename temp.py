# -*- coding: utf-8 -*- 
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf

"""TO clear defualt graph and nodes"""
tf.reset_default_graph()

with tf.name_scope('name_scope_x'):
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    'name_scope的get_var的scope不属于name_scope,属于name_scope的上一层'
    assert 'var1:0'== (var1.name) 

    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    'name_scope的var的scope，属于name_scope'
    assert 'name_scope_x/var2:0'== (var2.name)
    
    var3 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    'name_scope的var，自动增加了别名'
    assert 'name_scope_x/var2_1:0'== (var3.name) 


'name_scope不能设置reuse=True'
'name_scope和variable_scope都会给name加别名，所以这个name=name_scope_x_1'

with tf.name_scope('name_scope_x') as scope1:
    '!!!!!!if name=var1, will be conflict，即使类型不同也不行'
    var1 = tf.get_variable(name='var11', shape=[1], dtype=tf.int32)
    assert 'var11:0'== (var1.name)

    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    'name=name_scope_x_1，scope_name加别名了'
    assert 'name_scope_x_1/var2:0'== (var2.name)

    var3 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    assert 'name_scope_x_1/var2_1:0'== (var3.name) 


with tf.variable_scope('name_scope_x') as scope2:
    var1_v = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    assert 'name_scope_x/var1:0'== (var1_v.name)

    var2_v = tf.get_variable(name='var2', shape=[2], dtype=tf.float32)
    '这里看出，对get，name_scope/variable_scope，的name_scope应该是相通的，所以这里是var2_2'
    assert 'name_scope_x/var2_2:0'== (var2_v.name)

    var3_v = tf.Variable([0.52], name='var2')
    'name_scope别名了，因为这个scope自己本身的name就是name_scope_x_2'
    assert 'name_scope_x_2/var2:0'== (var3_v.name)
    
    'v_scope_1/bias_1'
    var_4 = tf.Variable([0.52], name='var4')
    assert 'name_scope_x_2/var4:0'== (var_4.name)

with tf.variable_scope('name_scope_x') as scope3:
    var1_vv = tf.Variable([0.52], name='var1')
    '无论上面有没有在name_scope_x_2声明变量，这个scope在创建的时候name就是name_scope_x_3'
    assert 'name_scope_x_3/var1:0'== (var1_vv.name)
    

'这个scope1，已经覆盖了前一个name_scope类型的scope1'
with tf.variable_scope('v_scope') as scope1:
    Weights1 = tf.get_variable('var1', shape=[2,3])
    assert 'v_scope/var1:0'== (Weights1.name)
    Scores1 = tf.get_variable('Scores', shape=[2,3])
    assert 'v_scope/Scores:0'== (Scores1.name)
    
    Bias1 = tf.Variable([0.52], name='Bias')
    assert 'v_scope/Bias:0'== (Bias1.name)

'reuse=True can reuse all get_variable in v_scope, including Weights and Scores'
'Attention, only get_variable, see bias2 bellowing'
with tf.variable_scope('v_scope', reuse=True) as scope5:
    Weights2 = tf.get_variable('var1')
    assert 'v_scope/var1:0'== (Weights2.name)
    Scores2 = tf.get_variable('Scores')
    assert 'v_scope/Scores:0'== (Scores2.name)
    'variable_scope里，不能get一个该scope里非get的变量。name_scope可以,会直接创建'
#    Bias2 = tf.get_variable('Bias')

    Bias3 = tf.Variable([0.52], name='Bias') 
    'variable_scope自动对name，进行了别名'
    assert 'v_scope_1/Bias:0'== (Bias3.name)
    Bias4 = tf.Variable([0.52], name='Bias')
    assert 'v_scope_1/Bias_1:0'== (Bias4.name)

""""""
"""Above is about one  -level Scope """
"""Below is about multi-level Scope """
""""""

with tf.variable_scope("foo") as foo_scope:
    assert foo_scope.name == "foo"
    
with tf.variable_scope("bar"):
    with tf.variable_scope("baz") as other_scope:
        assert other_scope.name == "bar/baz"
        '下面这2个例子就有意思了，自己领会'
        with tf.variable_scope(foo_scope) as foo_scope2:
            assert foo_scope2.name == "foo"
        with tf.variable_scope("foo") as foo_scope3:
            assert foo_scope3.name == "bar/baz/foo"

""""""
with tf.name_scope('nsc1'):
    v1 = tf.Variable([1], name='v1')
    with tf.variable_scope('vsc1'):
        v2 = tf.Variable([1], name='v2')
        v3 = tf.get_variable(name='v3', shape=[])
        assert ('nsc1/v1:0'== v1.name)
        assert( 'nsc1/vsc1/v2:0'==v2.name)
        '参见line-14：name_scope的get_var的scope不属于name_scope,属于name_scope的上一层'
        assert( 'vsc1/v3:0'==v3.name)

with tf.variable_scope("fool"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        x = 1.0 + v
        assert v.name == "fool/v:0"
        '这个和line_115的对比，name_scope能管住op的scope，管不住var的scope'
        assert x.op.name == "fool/bar/add"

""""""
""""""
print ("\n ----analysi end---- \n")