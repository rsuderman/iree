import iree.compiler.core as iree_mlir_compile
import iree.compiler.tflite as iree_tflite_compile
import iree.runtime as iree_rt
import numpy as np
import os as os
import re
import tempfile
import tensorflow.compat.v2 as tf

quantized = False

shape_x = (1, 12, 12, 4)
shape_y = (3, 3, shape_x[-1], 128 if quantized else 4)

class SimpleModule(tf.Module):
  def __init__(self):
    super(SimpleModule, self).__init__()
    self.y = np.random.rand(*shape_y).astype(np.single)

  @tf.function(
    input_signature=[
      tf.TensorSpec(shape=shape_x, dtype=tf.float32),
  ])
  def main(self, x):
    conv1 = tf.nn.depthwise_conv2d(x, self.y, strides=(1, 1, 1, 1), padding='VALID')
    return conv1

def compile_iree(tflite_path):
  workdir = os.path.dirname(tflite_path)
  tflite_ir = '/'.join([workdir, 'tflite.mlir'])
  imported_file = '/'.join([workdir, 'imported.mlir'])
  fixed_imported_file = '/'.join([workdir, 'fixed_imported.mlir'])
  binary_file = '/'.join([workdir, 'model.bytecode.hand'])

  print("Importing TFLite binary")  
  # Compile result to an imported form.
  iree_tflite_compile.compile_file(
    tflite_path, input_type="tosa", output_file=imported_file,
    save_temp_tfl_input=tflite_ir,
    target_backends=iree_tflite_compile.DEFAULT_TESTING_BACKENDS,
    import_only=True)

  # Insert iree.module.export to the main function.
  print("Adding iree.module.export")
  with open(imported_file, 'r') as f:
    pattern = re.compile(r"(func @main[ -~]*)(} {\n)")
    contents = f.read()
    match = pattern.search(contents)
    mainDecl = match.group(0)[:-4]
    newMain = mainDecl + ", iree.module.export"
    newContents = contents.replace(mainDecl, newMain)
    with open(fixed_imported_file, 'w') as of:
      of.write(newContents)

  print("Compiling for: ", ", ".join(iree_tflite_compile.DEFAULT_TESTING_BACKENDS))
  iree_mlir_compile.compile_file(fixed_imported_file, input_type="tosa", output_file=binary_file,
    target_backends=iree_tflite_compile.DEFAULT_TESTING_BACKENDS)

  return binary_file

def open_iree(iree_binary):
  with open(iree_binary, 'rb') as f:
    config = iree_rt.Config("dylib")
    ctx = iree_rt.SystemContext(config=config)
    vm_module = iree_rt.VmModule.from_flatbuffer(f.read())
    ctx.add_vm_module(vm_module)
    return ctx.modules.module["main"]


def main():
  simple_module = SimpleModule()
  with tempfile.TemporaryDirectory() as workdir:
    workdir = "/tmp/faketemp"
    module_path = "/".join([workdir, 'model.saved'])
    flatbuffer_path = '/'.join([workdir, 'model.flatbuffer'])

    tf.saved_model.save(simple_module, module_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(module_path)

    if quantized:
      def representative_data_gen():
        for i in range(100):
          sample = np.random.random(shape_x).astype(np.single)
          yield [sample]
      converter.representative_dataset = representative_data_gen

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(flatbuffer_path, 'wb') as f:
      f.write(tflite_model)

    ## Compiles the tflite model.
    tflite_interpreter = tf.lite.Interpreter(model_path=flatbuffer_path)
    tflite_interpreter.allocate_tensors()

    input_x = np.random.rand(*shape_x).astype(np.single)

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    tflite_interpreter.set_tensor(input_details[0]['index'], input_x)

    tflite_interpreter.invoke()
    tflite_results = np.array(tflite_interpreter.get_tensor(output_details[0]['index']))

    ## Compiles using iree
    iree_binary = compile_iree(flatbuffer_path)

    # Invoke IREE
    iree_invoke = open_iree(iree_binary)     
    iree_results = iree_invoke(input_x)

    compareTf2 = True
    error_bound = 1.0e-5
    if (compareTf2):
      # Invoke TF 2.0
      tf_results = np.array(simple_module.main(input_x))
      tflite_diff = np.abs(tflite_results - tf_results) < error_bound
      iree_diff = np.abs(iree_results - tf_results) < error_bound

      print("tflite match: {}".format(np.all(tflite_diff)))
      print("iree match: {}".format(np.all(iree_diff)))
    else:
      diff = np.abs(tflite_results - iree_results) < error_bound
      print("tflite-iree match: {}".format(np.all(diff)))
      print("tflite")
      print (tflite_results.reshape(output_shape[1:3]))
      print("iree")
      print (iree_results.reshape(output_shape[1:3]))

main()
