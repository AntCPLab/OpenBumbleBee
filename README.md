## BumbleBee: Secure Two-party Inference Framework for Large Transformers

This repo contains a proof-of-concept implementation for our paper.
The codes are still under heavy developments, and **should not** be used in any security sensitive product.

### Requirements

Our implementations are built on top of the  [SPU](https://github.com/secretflow/spu) library.

## Build

### Prerequisite

#### Linux

BumbleBee supports Linux build only.

```sh
Install gcc>=11.2, cmake>=3.18, ninja, nasm>=2.15, python==3.8, bazel==5.4.0, golang

python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
pip install 'transformers[flax]' #Install HuggingFace transformers library
```

Change the pointer activation functions to a Python call so that we can hijack them.

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

### Build

```sh
bazel build -c opt examples/python/...
```

It might take some times to fetch the dependencies.
    hidden_states = jax.nn.gelu(hidden_states)

## Run Private Inferernce

### Flax BERT Example

1. Install datasets

    ```sh
    pip install datasets
    ```

2. Launch SPU backend runtime

    ```sh
    env SPU_BB_SET_IEQUAL_BITS=16 bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/llm/flax_bert/2pc.json up
    ```

3. Run `flax_bert` example

    ```sh
    env SPU_BB_SET_IEQUAL_BITS=16 bazel run -c opt //examples/python/llm/flax_bert -- --config `pwd`/examples/python/llm/flax_bert/2pc.json
    ```

### Flax GPT2 Example

1. Launch SPU backend runtime

    ```sh
    env SPU_BB_SET_IEQUAL_BITS=16 bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/llm/flax_gpt2/2pc.json up
    ```

2. Run `flax_gpt2` example

    ```sh
    env SPU_BB_SET_IEQUAL_BITS=16 bazel run -c opt //examples/python/llm/flax_gpt2 -- --config `pwd`/examples/python/llm/flax_gpt2/2pc.json
    ```

### Flax VIT Example

1. Install packages

    ```sh
    pip install datasets
    ```

2. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/llm/flax_vit/2pc.json up
    ```

3. Run `flax_vit_inference` example

    ```sh
    bazel run -c opt //examples/python/llm/flax_vit -- --config `pwd`/examples/python/llm/flax_vit/2pc.json
    ```
