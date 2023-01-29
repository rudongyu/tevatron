# T5 as bi-encoder

```
python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --output_dir model_t5_base_msmarco_2048x2_100e \
  --model_name_or_path t5-base \
  --tokenizer_name t5-base \
  --save_steps 1000 \
  --dataset_name Tevatron/msmarco-passage \
  --bf16 \
  --per_device_train_batch_size 256 \
  --train_n_passages 2 \
  --learning_rate 1e-3 \
  --q_max_len 16 \
  --p_max_len 128 \
  --num_train_epochs 100 \
  --logging_steps 10 \
  --negatives_x_device \
  --overwrite_output_dir \
  --grad_cache \
  --gc_q_chunk_size 32 \
  --gc_p_chunk_size 32
```

