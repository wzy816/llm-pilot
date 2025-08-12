# llama.cpp

## build on ubuntu

```bash

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# update cuda-toolkit to 12.8
apt-get install cmake curl libcurl4-openssl-dev

cmake -B build
cmake --build build --config Release

```

## use llama-server

```bash
# download fim model
huggingface-cli download ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF

# start server
# https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md
# to match llama-server --fim-qwen-7b-default https://github.com/ggml-org/llama.cpp/blob/56304069599f4dd9749d94b9bca2c2c65bb27c02/common/arg.cpp#L3148
./build/bin/llama-server -m /path/to/qwen2.5-coder-7b-q8_0.gguf --port 8012 --host 0.0.0.0 -ngl 99 -fa -ub 1024 -b 1024 -c 0 --cache-reuse 256 --slots --parallel 2 --device cuda0 --threads 6 -v

# set ecs safety rules to open port 8012
```

## test

```bash
curl -X GET http://[ip]:8012/health

curl -X GET http://121.41.174.202:8012/slots

curl -X POST \
    --url http://[ip]:8012/completion \
    --header "Content-Type: application/json" \
    --data '{"prompt": "Building a website can be done in 10 simple steps:","n_predict": 128}'
```

## vscode

``` bash
# install llama.cpp plugin
#  set endpoint to http://[ip]:8012
```
