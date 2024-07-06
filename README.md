## BumbleBee: Secure Two-party Inference Framework for Large Transformers

This repo contains a proof-of-concept implementation for our paper.
The codes are still under heavy developments, and **should not** be used in any security sensitive product.

### Requirements

Our implementations are built on top of the  [SPU](https://github.com/secretflow/spu) library, specifically on [this commit](https://github.com/secretflow/spu/tree/724d3d1c3c37e891fcd0493a6cff0bf4310eda70).
Particularly, we have made the following changes.

* Adding a dispatch from DotGeneral to Batched MatMul. [commit](https://github.com/AntCPLab/OpenBumbleBee/commit/9bf547a71b958465f7ed5a32b69ea6c87690e5d7).
* Adding an intrinsic dispatch to the activation functions. [commit](https://github.com/AntCPLab/OpenBumbleBee/commit/967ed5a524c3d72763dcb84955ab558bdcc6463b)

## Build

### Prerequisite

#### Linux (Ubuntu)

We prefer a Linux build.

```sh
Install gcc>=11.2, cmake>=3.26, ninja, nasm>=2.15, python>=3.9, bazelisk, xxd, lld

python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
pip install 'transformers[flax]' #Install HuggingFace transformers library
```

About the commands used to install the above dependencies, you can follow [Ubuntu docker file](https://github.com/secretflow/devtools/blob/main/dockerfiles/ubuntu-base-ci.DockerFile).

### Change the pointer activation functions to a Python call so that we can hijack them.

**NOTE: Make sure the following modifications are properly done.**

**Why?** In SPU, we apply a Python hijack like [this](examples/python/ml/flax_bert/flax_bert.py#68) to replace the entire activation calls with our MPC protocols.
This hijack provides a simpler alternative to IR rewriting. Pattern matching all IR operations for complex activation functions, such as GeLU, is a tedious task. For the softmax function, we can easily apply this Python hijack. However, for the GeLU/SILU activations, we unfortunately need to modify the HuggingFace source code. This is because the activation calls used by HuggingFace are not Python calls but C pointer functions.

```python
# transformers/models/gpt2/modeling_flax_gpt2.py#294
def __call__(self, hidden_states, deterministic: bool = True):
    hidden_states = self.c_fc(hidden_states)
    #hidden_states = self.act(hidden_states)  
    # change the act pointer to a JAX call
    hidden_states = jax.nn.gelu(hidden_states)
    hidden_states = self.c_proj(hidden_states)
    hidden_states = self.dropout(hidden_states, deterministic=deterministic)
    return hidden_states

# transformers/models/bert/modeling_flax_bert.py#465
def __call__(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    #hidden_states = self.activation(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    return hidden_states

# transformers/models/vit/modeling_flax_vit.py#278
def __call__(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    #hidden_states = self.activation(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    return hidden_states
```

Also, we can skip some computation in JAX's argmax function. 
This can reduce some cost in the one-hot encoding (e.g., embedding lookup).

```python
#jax/_src/lax/lax.py#3919
def _compute_argminmax(value_comparator, get_identity,
                       operand, *, index_dtype, axes):
  # value_comparator is either lax.lt (for argmin) or lax.gt
  # get_identity(operand.dtype) is inf for argmin or -inf for argmax
  axis, = axes
  indices = broadcasted_iota(index_dtype, np.shape(operand), axis)
  def reducer_fn(op_val_index, acc_val_index):
    op_val, op_index = op_val_index
    acc_val, acc_index = acc_val_index
    # NOTE(lwj): we skip the NaN check in ArgMax which is meaningless in MPC
    pick_op = value_comparator(op_val, acc_val)
    return (select(pick_op, op_val, acc_val),
            select(pick_op, op_index, acc_index))
    #return (select(pick_op_val, op_val, acc_val),
    # Pick op_val if Lt (for argmin) or if NaN
    #pick_op_val = bitwise_or(value_comparator(op_val, acc_val),
    #                         ne(op_val, op_val))
    # If x and y are not NaN and x = y, then pick the first
    #pick_op_index = bitwise_or(pick_op_val,
    #                           bitwise_and(eq(op_val, acc_val),
    #                                       lt(op_index, acc_index)))
    #return (select(pick_op_val, op_val, acc_val),
    #        select(pick_op_index, op_index, acc_index))
```

### Build Main Programs 

```sh
bazel build -c opt examples/python/ml/flax_gpt2/...
bazel build -c opt examples/python/ml/flax_bert/...
bazel build -c opt examples/python/ml/flax_vit/...
```

It might take some times to fetch the dependencies.

## Run Microbenchmarks

* `bazel run -c opt examples/python/microbench:gelu`

* `bazel run -c opt examples/python/microbench:softmax`

* `bazel run -c opt examples/python/microbench:matmul`


## Run Private Inferernce

### Flax BERT Example

1. Install datasets

    ```sh
    pip install datasets
    ```

2. Launch SPU backend runtime

    ```sh
    env SPU_BB_SET_IEQUAL_BITS=15 bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/conf/2pc.json up
    ```

3. Run `flax_bert` example

    ```sh
    env SPU_BB_SET_IEQUAL_BITS=15 bazel run -c opt //examples/python/ml/flax_bert -- --config `pwd`/examples/python/conf/2pc.json
    ```

We prefer to set the env variable `SPU_BB_SET_IEQUAL_BITS` equals to the vocabulary size. For instance, in BERT-base, we have about 30522 vocabulary.
Thus `SPU_BB_SET_IEQUAL_BITS=15` would be enough for the one-hot computation.

### Flax GPT2 Example

1. Launch SPU backend runtime

    ```sh
    env SPU_BB_SET_IEQUAL_BITS=16 bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/conf/2pc.json up
    ```

2. Run `flax_gpt2` example

    ```sh
    env SPU_BB_SET_IEQUAL_BITS=16 bazel run -c opt //examples/python/ml/flax_gpt2 -- --config `pwd`/examples/python/conf/2pc.json
    ```
The vocabulary size in GPT2 is about 50k. Thus we need 16bits for the one-hot computation.

### Flax VIT Example

1. Install packages

    ```sh
    pip install datasets
    ```

2. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/conf/2pc.json up
    ```

3. Run `flax_vit` example

    ```sh
    bazel run -c opt //examples/python/ml/flax_vit/flax_vit -- --config `pwd`/examples/python/conf/2pc.json
    ```
