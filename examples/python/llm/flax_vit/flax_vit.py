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

# Reference: https://huggingface.co/docs/transformers/model_doc/vit#transformers.FlaxViTForImageClassification
from transformers import AutoImageProcessor, FlaxViTForImageClassification
from PIL import Image
import jax.nn as jnn
import jax.numpy as jnp
import argparse, json
import requests
import jax
import time
import flax.linen as fnn
import spu.utils.distributed as ppd
import spu.intrinsic as intrinsic
import spu.utils.distributed as ppd
import spu.spu_pb2 as spu_pb2
from contextlib import contextmanager

copts = spu_pb2.CompilerOptions()
# enable x / broadcast(y) -> x * broadcast(1/y) which accelerate the softmax
copts.enable_optimize_denominator_with_broadcast = True

def parse_args():
    parser = argparse.ArgumentParser(description="distributed driver.")
    parser.add_argument(
        "-c", "--config", default="examples/python/ml/flax_vit/2pc-local.json"
    )
    return parser.parse_args()


args = parse_args()
with open(args.config, "r") as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


@contextmanager
def hijack(msg: str, enabled=True):
    if not enabled:
        yield
        return
    # hijack some target functions
    jnn_gelu = jnn.gelu
    fnn_gelu = fnn.gelu
    jnn.gelu = intrinsic.f_gelu
    fnn.gelu = intrinsic.f_gelu
    yield
    # recover back
    jnn.gelu = jnn_gelu
    fnn.gelu = fnn_gelu


def run_on_cpu(model, inputs):
    print(f"Running on CPU ...")
    params = model.params

    def eval(params, inputs):
        outputs = model(inputs, params=params)
        return outputs.logits

    start = time.time()
    logits = eval(params, inputs)
    predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
    end = time.time()
    print(f"CPU runtime: {(end - start)}s")
    print("CPU Predicted class:", model.config.id2label[predicted_class_idx.item()])

def run_on_spu(model, inputs):
    print(f"Running on SPU ...")
    params = model.params

    def eval(params, inputs):
        with hijack("gelu", enabled=True):
            outputs = model(inputs, params=params)
        return outputs.logits

    inputs = ppd.device("P1")(lambda x: x)(inputs)
    params = ppd.device("P1")(lambda x: x)(params)

    start = time.time()
    logits_spu = ppd.device("SPU")(eval, copts=copts)(params, inputs)
    predicted_class_idx = jax.numpy.argmax(ppd.get(logits_spu), axis=-1)
    end = time.time()
    print(f"SPU runtime: {(end - start)}s")
    print("SPU Predicted class:", model.config.id2label[predicted_class_idx.item()])

def main():
    # load dataset
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # the pre-processor is supposed to be public
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    inputs = image_processor(images=image, return_tensors="np")["pixel_values"]

    # load models
    model = FlaxViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    outputs = model(inputs)
    logits = outputs.logits

    # model predicts one of the 1000 ImageNet classes
    print("Inference ...")
    run_on_cpu(model, inputs)
    run_on_spu(model, inputs)


if __name__ == "__main__":
    main()
