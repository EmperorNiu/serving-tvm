import os
import numpy as np

import tvm
from tvm import relay
import os.path
import tarfile,sys

# Tensorflow imports

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
import tvm.relay.testing.tf as tf_testing

def untar(fname):
    file_tar, file_tar_ext = os.path.splitext(fname)
    print(file_tar)
    if (fname.endswith("tgz")):
        tar = tarfile.open(fname)
        tar.extractall(path="./" + file_tar)
        tar.close()
        print("Extracted in Current Directory")
    else:
        print("Not a tar.gz file")


def get_workload(path):
    from mxnet.gluon.utils import download
    download(path, ".")

    tar_name = os.path.basename(path)
    untar(tar_name)

    file_tar, file_tar_ext = os.path.splitext(tar_name)
    model_name = file_tar + "/" + file_tar + "_frozen.pb"
    return model_name

name = 'mobilenet_v1_1.0_224'
dload_path ='http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/' + name +'.tgz'
model_name = get_workload(dload_path)



def import_into_tvm(graph_def, input_data, input_node, num_output=1):
    """ Generic function to compile on relay and execute on tvm """

    shape_dict = {input_node: input_data.shape}
    dtype_dict = {input_node: input_data.dtype}

    sym, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)
    return sym, params


ops.reset_default_graph()
with tf.gfile.FastGFile(os.path.join("./", model_name), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)

    in_shape = (1, 224, 224 , 3)    
    shape_dict = {'input': in_shape}
    dtype_dict = {'input': "float32"}

    sym, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict)
        
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(
            sym, target="llvm", params=params)

        lib.export_library("model.so")
        with open("model.json", "w") as fo:
            fo.write(graph)
        with open("model.params", "wb") as fo:
            # import nnvm
            fo.write(relay.save_param_dict(params))

