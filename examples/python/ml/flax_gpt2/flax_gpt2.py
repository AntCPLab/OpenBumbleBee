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
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/conf/2pc.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/flax_gpt2 -- --config `pwd`/examples/python/conf/2pc.json


import argparse
import json
import os
import time
from contextlib import contextmanager

import flax.linen as fnn
import jax
import jax.nn as jnn
from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, GPT2Config

import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd

copts = spu_pb2.CompilerOptions()
# enable x / broadcast(y) -> x * broadcast(1/y) which accelerate the softmax
copts.enable_optimize_denominator_with_broadcast = True

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/ml/flax_gpt2/2pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


def _gelu(x):
    return intrinsic.spu_gelu(x)


def _softmax(x, axis=-1, where=None, initial=None):
    x_max = jax.numpy.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max
    # spu.neg_exp will clip values that too large.
    # nexp = jax.numpy.exp(x) * (x > -14.0)
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


TOKEN_NUM = 8


def run_on_cpu(model, input_ids, tokenizer):
    print(f"Running on CPU ...")
    params = model.params

    # greedy search
    # ref: https://huggingface.co/blog/how-to-generate
    def eval(params, input_ids, token_num=TOKEN_NUM):
        for _ in range(token_num):
            outputs = model(input_ids=input_ids, params=params)
            next_token_logits = outputs[0][0, -1, :]
            next_token = jax.numpy.argmax(next_token_logits)
            input_ids = jax.numpy.concatenate(
                [input_ids, jax.numpy.array([[next_token]])], axis=1
            )
        return input_ids

    start = time.time()
    output_ids = eval(params, input_ids)
    end = time.time()
    output_tokens = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"CPU runtime: {(end - start)}s\noutput {output_tokens}")


def run_on_spu(model, input_ids, tokenizer):
    print(f"Running on SPU ...")
    params = model.params

    def eval(params, input_ids, token_num=TOKEN_NUM):
        for _ in range(token_num):
            with hijack(enabled=True):
                outputs = model(input_ids=input_ids, params=params)
            next_token_logits = outputs[0][0, -1, :]
            next_token = jax.numpy.argmax(next_token_logits)
            input_ids = jax.numpy.concatenate(
                [input_ids, jax.numpy.array([[next_token]])], axis=1
            )
        return input_ids

    spu_input_ids = ppd.device("P1")(lambda x: x)(input_ids)
    spu_params = ppd.device("P2")(lambda x: x)(params)
    start = time.time()
    outputs_ids_spu = ppd.device("SPU")(eval, copts=copts)(spu_params, spu_input_ids)
    end = time.time()
    output_ids = ppd.get(outputs_ids_spu)
    output_tokens = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"SPU runtime: {(end - start)}s\noutput {output_tokens}")


def main(tokenizer_func, model_func, checkpoint):
    model = model_func.from_pretrained(checkpoint)
    tokenizer = tokenizer_func.from_pretrained(checkpoint)
    input_ids = tokenizer.encode(
        'I enjoy walking with my cute dog', return_tensors='jax'
    )

    run_on_cpu(model, input_ids, tokenizer)
    run_on_spu(model, input_ids, tokenizer)


if __name__ == '__main__':
    tokenizer = AutoTokenizer
    model = FlaxGPT2LMHeadModel
    checkpoint = "gpt2"

    main(tokenizer, model, checkpoint)
