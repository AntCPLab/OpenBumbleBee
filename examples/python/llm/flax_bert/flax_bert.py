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
import json
import time
from contextlib import contextmanager

import flax.linen as fnn
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from flax.linen.linear import Array
from transformers import (
    BertConfig,
    BertTokenizerFast,
    FlaxBertForSequenceClassification,
    FlaxRobertaForSequenceClassification,
    RobertaTokenizerFast,
)

import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd

copts = spu_pb2.CompilerOptions()
# enable x / broadcast(y) -> x * broadcast(1/y) which accelerate the softmax
copts.enable_optimize_denominator_with_broadcast = True


def parse_args():
    parser = argparse.ArgumentParser(description="distributed driver.")
    parser.add_argument(
        "-c", "--config", default="examples/python/ml/flax_bert/2pc-local.json"
    )
    parser.add_argument("--model", default="bert", type=str)
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


def run_on_cpu(model, input_ids, attention_masks, labels):
    print(f"Running on CPU ...")
    params = model.params

    def eval(params, input_ids, attention_masks):
        logits = model(input_ids, attention_masks, params=params)[0]
        return logits

    start = time.time()
    logits = eval(params, input_ids, attention_masks)
    end = time.time()
    print(f"CPU runtime: {(end - start)}s\noutput logits: {logits}")


def run_on_spu(model, input_ids, attention_masks, labels):
    print(f"Running on SPU ...")
    params = model.params

    def eval(params, input_ids, attention_masks):
        with hijack("gelu", enabled=True):
            logits = model(input_ids, attention_masks, params=params)[0]
        return logits

    spu_input_ids = ppd.device("P1")(lambda x: x)(input_ids)
    spu_attention_masks = ppd.device("P1")(lambda x: x)(attention_masks)
    spu_params = ppd.device("P1")(lambda x: x)(params)
    start = time.time()
    logits_spu = ppd.device("SPU")(eval, copts=copts)(
        spu_params, spu_input_ids, spu_attention_masks
    )
    end = time.time()
    logits_spu = ppd.get(logits_spu)
    print(f"SPU runtime: {(end - start)}s\noutput logits: {logits_spu}")


def main(tokenizer_func, model_func, checkpoint):
    dataset = load_dataset("glue", "cola", split="test")
    model = model_func.from_pretrained(checkpoint)
    tokenizer = tokenizer_func.from_pretrained(checkpoint)

    for dummy_input in dataset:
        features, labels = dummy_input["sentence"], dummy_input["label"]

        input_ids, attention_masks = (
            tokenizer(
                features,
                return_tensors="jax",
            )["input_ids"],
            tokenizer(
                features,
                return_tensors="jax",
            )["attention_mask"],
        )

        run_on_cpu(model, input_ids, attention_masks, labels)
        run_on_spu(model, input_ids, attention_masks, labels)
        break  # just test one sentense


if __name__ == "__main__":
    tokenizer = BertTokenizerFast
    model = FlaxBertForSequenceClassification
    checkpoint = "bert-base-uncased"
    main(tokenizer, model, checkpoint)
