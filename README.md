# llama2_from_scrach


```
# install libraries
uv venv llama --python 3.11
uv pip install -r requirements.txt
source env.sh

# download pretrained checkpoint
bash download.sh

# run
python inference.py
```

## Acknowledgements
This implementation follows the YouTube tutorial [Coding LLaMA 2 from scratch in PyTorch - KV Cache, Grouped Query Attention, Rotary PE, RMSNorm](https://www.youtube.com/watch?v=oM4VmoabDAI)