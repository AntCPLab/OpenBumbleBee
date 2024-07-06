# Bumblebee's Flax VIT Example

This example demonstrates how to use SPU to perform inference and fine-tune based on the [ViT](https://arxiv.org/abs/2010.11929) model privately.

This example comes from Hugging Face x Flax official github repo:

<https://huggingface.co/docs/transformers/model_doc/vit>

1. Install packages

    ```sh
    pip install -r requirements.txt
    ```

2. Launch SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- -c `pwd`/examples/python/conf/2pc.json up
    ```

3. Run `flax_vit_inference` example

    ```sh
    bazel run -c opt //examples/python/ml/flax_vit:flax_vit_inference -- --config `pwd`/examples/python/conf/2pc.json

    ```
