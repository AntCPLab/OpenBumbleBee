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

import argparse
import time
import json
import jax.numpy as jnp
import jax.nn as jnn
import flax.linen as fnn
from flax.linen.linear import Array
from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, GPT2Config

from contextlib import contextmanager

import spu.intrinsic as intrinsic
import spu.utils.distributed as ppd
import spu.spu_pb2 as spu_pb2


def parse_args():
    parser = argparse.ArgumentParser(description="distributed driver.")
    parser.add_argument(
        "-c", "--config", default="examples/python/ml/flax_gpt2/2pc-local.json"
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


tokenizer = AutoTokenizer.from_pretrained("gpt2")
pretrained_model = FlaxGPT2LMHeadModel.from_pretrained("gpt2")

copts = spu_pb2.CompilerOptions()
# enable x / broadcast(y) -> x * broadcast(1/y) which accelerate the softmax
copts.enable_optimize_denominator_with_broadcast = True

# greedy search
# ref: https://huggingface.co/blog/how-to-generate
token_num = 10


def text_generation_cpu(input_ids, params, token_num=token_num):
    config = GPT2Config(activation_function="gelu_new")
    model = FlaxGPT2LMHeadModel(config=config)
    for _ in range(token_num):
        outputs = model(input_ids=input_ids, params=params)
        next_token_logits = outputs[0][0, -1, :]
        next_token = jnp.argmax(next_token_logits)
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
    return input_ids


def text_generation(input_ids, params, token_num=token_num):
    config = GPT2Config(activation_function="gelu_new")
    model = FlaxGPT2LMHeadModel(config=config)

    for _ in range(token_num):
        with hijack("gelu", enabled=True):
            outputs = model(input_ids=input_ids, params=params)
        next_token_logits = outputs[0][0, -1, :]
        next_token = jnp.argmax(next_token_logits)
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
    return input_ids


def run_on_cpu():
    # encode context the generation is conditioned on
    inputs_ids = tokenizer.encode("I enjoy walking with my dog", return_tensors="jax")
    start = time.time()
    outputs_ids = text_generation_cpu(inputs_ids, pretrained_model.params)
    end = time.time()
    print(f"CPU runtime: {(end - start)}s")
    return outputs_ids


def run_on_spu():
    # encode context the generation is conditioned on
    inputs_ids = tokenizer.encode("I enjoy walking with my dog", return_tensors="jax")
    input_ids = ppd.device("P1")(lambda x: x)(inputs_ids)
    params = ppd.device("P2")(lambda x: x)(pretrained_model.params)

    start = time.time()
    outputs_ids = ppd.device("SPU")(text_generation, copts=copts)(input_ids, params)
    outputs_ids = ppd.get(outputs_ids)
    end = time.time()
    print(f"SPU runtime: {(end - start)}s")
    return outputs_ids


if __name__ == "__main__":
    print("\n------\nRun on CPU")
    outputs_ids = run_on_cpu()
    print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
    print("\n------\nRun on SPU")
    outputs_ids = run_on_spu()
    print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
