# Flax VIT Example

## Config

We describe the config file `flax_vit/2pc-local.json`.

```json
"runtime_config": {
    "protocol": "CHEETAH", # MPC back-end
    "field": "FM64",       # secret sharing on 64-bit
    "enable_pphlo_profile": true,
    "enable_hal_profile": true,
    "enable_pphlo_trace": false,
    "enable_action_trace": false,
    "experimental_disable_mmul_split": true,
    "fxp_exp_mode": 0,  # use Taylor expansion for exp
    "fxp_exp_iters": 6, # degree-6 Taylor expansion
    "enable_heurisitc_truncate": true,      # Use Keller's heuristic for efficient truncate
    "allow_high_prob_one_bit_error": true,  # 1-bit error bOLEe
    "experimental_enable_batch_mmul": true, # compress batched matmul
    "experimental_i_equal_bits": 16         # GPT2 vocab size is about 50k. Thus i_equal on 16-bit is enough for the token-id to one-hot
}
```

This example demonstrates how to use SPU to perform inference and fine-tune based on the [ViT](https://arxiv.org/abs/2010.11929) model privately.

This example comes from Hugging Face x Flax official github repo:

<https://huggingface.co/docs/transformers/model_doc/vit>

1. Install packages

    ```sh
    pip install -r requirements.txt
    ```

2. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_vit/2pc-local.json up
    ```

3. Run `flax_vit_inference` example

    ```sh
    bazel run -c opt //examples/python/ml/flax_vit:flax_vit_inference -- --config `pwd`/examples/python/ml/flax_vit/2pc-local.json
    ```

