# Lint as: python3
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test all vision models from slim lib."""

from absl import app
from absl import flags
import iree.compiler.core as iree_mlir_compile
import iree.compiler.tflite as iree_tflite_compile
import iree.runtime as iree_rt
from iree.tools import tflite as iree_tflite
import numpy as np
import pathlib
import re
import tempfile
import urllib.request

FLAGS = flags.FLAGS

models = {
  "fastspeech" : "https://tfhub.dev/tulasiram58827/lite-model/fastspeech2/dr/1?lite-format=tflite",
  "mobile_bert" : "https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite",
  "magenta" : "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/fp16/transfer/1?lite-format=tflite",
  "mobilenet_v1" : "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_160/1/default/1?lite-format=tflite",
  "mobilenet_v2" : "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/3?lite-format=tflite",
  "mobilenet_v3" : "https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_large_100_224/feature_vector/5/default/1?lite-format=tflite",
  "posenet" : "https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite",
  "german_tacotron" : "https://tfhub.dev/monatis/lite-model/german-tacotron2/lite/1?lite-format=tflite",
  "gan" : "https://tfhub.dev/captain-pool/lite-model/esrgan-tf2/1?lite-format=tflite",
  "yamnet": "https://tfhub.dev/google/lite-model/yamnet/tflite/1?lite-format=tflite",
}

# Testing vision models from
# https://github.com/tensorflow/models/tree/master/research/slim
# slim models were designed with tf v1 and then coverted to SavedModel
# they are stored at tensorflow_hub.
flags.DEFINE_string('model',
    'mobile_bert',
    'string representing the model to downloads:\n' +
    ", ".join(models.keys()))

TFLITE_DOWNLOAD_POST = '/1?lite-format=tflite'


def ConvertDeclToSignature(decl):
  pattern = re.compile(r"arg\d: tensor<[a-zA-Z0-9]*>")
  match = pattern.findall(decl)
  signature = []
  for s in match:
    parts = s[13:-1].split('x')
    shape = [int(n) for n in parts[:-1]]
    type_str = parts[-1]
    if (type_str == "i8"):
      typ = np.int8
    if (type_str == "i16"):
      typ = np.int16
    if (type_str == "i32"):
      typ = np.int32
    elif (type_str == "f32"):
      typ = np.single
    else:
      print("Error: unknown type: ", type_str)
      exit()

    signature.append((shape, typ))
  return signature

def ExtractSignatureFromMlir(mlir_file):
  with open(mlir_file, 'r') as f:
    pattern = re.compile(r"(func @main[ -~]*)(} {\n)")
    contents = f.read()
    match = pattern.search(contents)
    decl = match.group(0)
    return ConvertDeclToSignature(decl)
    

def main(argv):
  del argv  # Unused.
  model_path = models[FLAGS.model]

  with tempfile.TemporaryDirectory() as workdir:
    workdir = "/tmp/fake_temp_dir"
    print("Workdir: ", workdir)

    tflite_file = '/'.join([workdir, 'model.tflite'])
    tflite_ir = '/'.join([workdir, 'tflite.mlir'])
    imported_file = '/'.join([workdir, 'imported.mlir'])
    fixed_imported_file = '/'.join([workdir, 'fixed_imported.mlir'])
    binary_file = '/'.join([workdir, 'model.bytecode.hand'])

    print ("Downloading: ", model_path)
    urllib.request.urlretrieve(model_path, tflite_file)
    print("Size (kb):", int(pathlib.Path(tflite_file).stat().st_size/1024))  


    print("Importing TFLite binary")  
    # Compile result to an imported form.
    iree_tflite_compile.compile_file(
      tflite_file, input_type="tosa", output_file=imported_file,
      save_temp_tfl_input=tflite_ir,
      target_backends=iree_tflite_compile.DEFAULT_TESTING_BACKENDS, import_only=True)


    # Detect what the input args are.
    signature = ExtractSignatureFromMlir(imported_file)

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

    print("Using binary: ", iree_tflite.get_tool('iree-import-tflite'))
    with open(binary_file, 'rb') as f:
      config = iree_rt.Config("dylib")
      ctx = iree_rt.SystemContext(config=config)
      vm_module = iree_rt.VmModule.from_flatbuffer(f.read())
      ctx.add_vm_module(vm_module)

      print(signature)

      args = []
      for arg in signature:
        (shape, ty) = arg
        args.append(np.zeros(shape, dtype=ty))

      invoke = ctx.modules.module["main"]
      invoke(*args)

if __name__ == '__main__':
  app.run(main)
