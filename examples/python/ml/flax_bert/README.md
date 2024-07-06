# Bumblebee's Flax BERT Example 

1. Install huggingface transformers library

    ```sh
    pip install 'transformers[flax]'
    ```

2. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/conf/2pc.json up
    ```

3. Run `flax_gpt2` example

    ```sh
    bazel run -c opt //examples/python/ml/flax_bert -- --config `pwd`/examples/python/conf/2pc.json

    ```
