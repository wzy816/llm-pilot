# gpt-oss vllm

```bash
conda env remove -n vllm.gpt-oss

conda create -n vllm.gpt-oss python=3.12

source activate vllm.gpt-oss

pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

vllm serve openai/gpt-oss-20b
```
