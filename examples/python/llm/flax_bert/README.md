# Flax BERT Example
## Config

We describe the config file `flax_bert/2pc-local.json`.

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
    "experimental_i_equal_bits": 15         # BERT vocab size is about 30k. Thus i_equal on 15-bit is enough for the token-id to one-hot
}
```

This example demonstrates how to use SPU to run private inference on a pre-trained BERT model.

1. Install huggingface transformers library

    ```sh
    pip install 'transformers[flax]'
    ```

2. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_bert/2pc-local.json up
    ```

3. Run `flax_bert` example

    ```sh
    bazel run -c opt //examples/python/ml/flax_bert -- --config `pwd`/examples/python/ml/flax_bert/2pc-local.json
    ```
