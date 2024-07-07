## BumbleBee: Secure Two-party Inference Framework for Large Transformers

This repo contains a proof-of-concept implementation for our paper.
The codes are still under heavy developments, and **should not** be used in any security sensitive product.

### Requirements

Our implementations are built on top of the  [SPU](https://github.com/secretflow/spu) library, specifically on [this commit](https://github.com/secretflow/spu/tree/724d3d1c3c37e891fcd0493a6cff0bf4310eda70).
Particularly, we have made the following changes.

* Adding a dispatch from DotGeneral to Batched MatMul. [commit](https://github.com/AntCPLab/OpenBumbleBee/commit/9bf547a71b958465f7ed5a32b69ea6c87690e5d7).
* Adding an intrinsic dispatch to the activation functions. [commit](https://github.com/AntCPLab/OpenBumbleBee/commit/967ed5a524c3d72763dcb84955ab558bdcc6463b)

## Build

### 1. Prerequisite
We prefer a Linux build. The following build has been tested on **Ubuntu 22.04**. 

```bash
# set TARGETPLATFORM='linux/arm64' if ARM CPU is used.

# Update dependencies
apt update \
&& apt upgrade -y \
&& apt install -y gcc-11 g++-11 libasan6 \
git wget curl unzip autoconf make lld-15 \
cmake ninja-build vim-common libgl1 libglib2.0-0 \
&& apt clean \
&& update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
&& update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
&& update-alternatives --install /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-15 100 

# clang is required on arm64 platform
if [ "$TARGETPLATFORM" = "linux/arm64" ] ; then apt install -y clang-15 \
    && apt clean \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100 \
; fi


# amd64 is only reqiured on amd64 platform
if [ "$TARGETPLATFORM" = "linux/amd64" ] ; then apt install -y nasm ; fi

# install conda
if [ "$TARGETPLATFORM" = "linux/arm64" ] ; then CONDA_ARCH=aarch64 ; else CONDA_ARCH=x86_64 ; fi \
  && wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.3.0-0-Linux-$CONDA_ARCH.sh \
  && bash Miniconda3-py310_24.3.0-0-Linux-$CONDA_ARCH.sh -b \
  && rm -f Miniconda3-py310_24.3.0-0-Linux-$CONDA_ARCH.sh \
  && /root/miniconda3/bin/conda init

# Add conda to path
export PATH="/root/miniconda3/bin:${PATH}" 

# install bazel 
if [ "$TARGETPLATFORM" = "linux/arm64" ] ; then BAZEL_ARCH=arm64 ; else BAZEL_ARCH=amd64 ; fi \
  && wget https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-$BAZEL_ARCH \
  && mv bazelisk-linux-$BAZEL_ARCH /usr/bin/bazel \
  && chmod +x /usr/bin/bazel

# install python dependencies
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
```

About the commands used to install on other platform, you can follow [Ubuntu docker file](https://github.com/secretflow/devtools/blob/main/dockerfiles/).

### 2. Modify Some Python Codes

**Why?** In SPU, we apply a Python hijack like [this](examples/python/ml/flax_bert/flax_bert.py#L68) to replace the entire activation calls with our MPC protocols.
This hijack provides a simpler alternative to IR rewriting. Pattern matching all IR operations for complex activation functions, such as GeLU, is a tedious task. For the softmax function, we can easily apply this Python hijack. However, for the GeLU/SILU activations, we unfortunately need to modify the HuggingFace source code. This is because the activation calls used by HuggingFace are not Python calls but C pointer functions.

```python
# transformers/models/gpt2/modeling_flax_gpt2.py#297
def __call__(self, hidden_states, deterministic: bool = True):
    hidden_states = self.c_fc(hidden_states)
    # hidden_states = self.act(hidden_states)  
    hidden_states = jax.nn.gelu(hidden_states)
    hidden_states = self.c_proj(hidden_states)
    hidden_states = self.dropout(hidden_states, deterministic=deterministic)
    return hidden_states

# transformers/models/bert/modeling_flax_bert.py#468
def __call__(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    #hidden_states = self.activation(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    return hidden_states

# transformers/models/vit/modeling_flax_vit.py#282
def __call__(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    #hidden_states = self.activation(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    return hidden_states
```

Also, we can skip some computation in JAX's argmax function. 
This can reduce some cost in the one-hot encoding (e.g., embedding lookup).

```python
#jax/_src/lax/lax.py#4118
def __call__(self, op_val_index, acc_val_index):
     op_val, op_index = op_val_index
     acc_val, acc_index = acc_val_index
     pick_op_val = self._value_comparator(op_val, acc_val)
     return (select(pick_op_val, op_val, acc_val),
            select(pick_op_val, op_index, acc_index))
     # Pick op_val if Lt (for argmin) or if NaN
     #pick_op_val = bitwise_or(self._value_comparator(op_val, acc_val),
     #                          ne(op_val, op_val))
     # If x and y are not NaN and x = y, then pick the first
     #pick_op_index = bitwise_or(pick_op_val,
     #                           bitwise_and(eq(op_val, acc_val),
     #                           lt(op_index, acc_index)))
     #return (select(pick_op_val, op_val, acc_val),
     #       select(pick_op_index, op_index, acc_index))
```

### 3. Build the Main Programs 

```sh
bazel build -c opt examples/python/ml/flax_gpt2/...
bazel build -c opt examples/python/ml/flax_bert/...
bazel build -c opt examples/python/ml/flax_vit/...
```

It might take some times to fetch the dependencies.

```
INFO: Elapsed time: 403.615s, Critical Path: 277.54s
INFO: 3686 processes: 223 internal, 3463 linux-sandbox.
INFO: Build completed successfully, 3686 total actions
```

## Run Microbenchmarks

* `bazel run -c opt examples/python/microbench:gelu`

* `bazel run -c opt examples/python/microbench:softmax`

* `bazel run -c opt examples/python/microbench:matmul`


## Run Private Inferernce

**NOTE** We need to fetch models from HuggingFace. Please make sure your network condition :).

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
The vocabulary size in GPT2 is about 50k. Thus we set `SPU_BB_SET_IEQUAL_BITS=16` for the one-hot computation.

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
