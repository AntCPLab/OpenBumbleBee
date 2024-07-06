# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/flax_vit:flax_vit_inference

import argparse
import json
import os
import time
from contextlib import contextmanager

import flax.linen as fnn
import jax
import jax.nn as jnn
import requests
from PIL import Image

# Reference: https://huggingface.co/docs/transformers/model_doc/vit#transformers.FlaxViTForImageClassification
from transformers import AutoImageProcessor, FlaxViTForImageClassification

import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd

copts = spu_pb2.CompilerOptions()
# enable x / broadcast(y) -> x * broadcast(1/y) which accelerate the softmax
copts.enable_optimize_denominator_with_broadcast = True

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/2pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


def _gelu(x):
    return intrinsic.spu_gelu(x)


def _softmax(x, axis=-1, where=None, initial=None):
    x_max = jax.numpy.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max
    # exp on large negative is clipped to zero
    nexp = intrinsic.spu_neg_exp(x)
    divisor = jax.numpy.sum(nexp, axis, where=where, keepdims=True)
    return nexp / divisor


@contextmanager
def hijack(enabled=True):
    if not enabled:
        yield
        return
    # hijack some target functions
    jnn_gelu = jnn.gelu
    fnn_gelu = fnn.gelu
    jnn_sm = jnn.softmax
    fnn_sm = fnn.softmax

    jnn.gelu = _gelu
    fnn.gelu = _gelu
    jnn.softmax = _softmax
    fnn.softmax = _softmax

    yield
    # recover back
    jnn.gelu = jnn_gelu
    fnn.gelu = fnn_gelu
    jnn.softmax = jnn_sm
    fnn.softmax = fnn_sm


def run_on_cpu(model, inputs):
    print(f"Running on CPU ...")
    params = model.params

    def eval(params, inputs):
        outputs = model(inputs, params=params)
        return outputs.logits

    start = time.time()
    logits = eval(params, inputs)
    end = time.time()
    predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
    print(f"CPU runtime: {(end - start)}s")
    print("Top 5 logits ", logits[:, :5])
    print("CPU Predicted class:", model.config.id2label[predicted_class_idx.item()])


def run_on_spu(model, inputs):
    print(f"Running on SPU ...")
    params = model.params

    def eval(params, inputs):
        with hijack(enabled=True):
            outputs = model(inputs, params=params)
        return outputs.logits

    inputs = ppd.device("P1")(lambda x: x)(inputs)
    params = ppd.device("P2")(lambda x: x)(params)

    start = time.time()
    logits = eval(params, inputs)
    logits_spu = ppd.device("SPU")(eval, copts=copts)(params, inputs)
    end = time.time()
    predicted_class_idx = jax.numpy.argmax(ppd.get(logits_spu), axis=-1)
    print(f"SPU runtime: {(end - start)}s")
    print("Top 5 logits ", ppd.get(logits_spu)[:, :5])
    print("SPU Predicted class:", model.config.id2label[predicted_class_idx.item()])


def main():
    # load dataset
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # the pre-processor is supposed to be public
    image_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224",
        cache_dir='/Volumes/HUB/huggingface/hub',
        local_files_only=True,
    )

    inputs = image_processor(images=image, return_tensors="np")["pixel_values"]

    # load models
    model = FlaxViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        cache_dir='/Volumes/HUB/huggingface/hub',
        local_files_only=True,
    )

    run_on_cpu(model, inputs)
    run_on_spu(model, inputs)


if __name__ == "__main__":
    main()
