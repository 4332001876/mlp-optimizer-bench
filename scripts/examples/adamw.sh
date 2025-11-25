python -m mlp_optimizer_bench.run_optim_bench \
  --dataset high_order_cls \
  --n-features 50 \
  --hidden-dims 256,256,256 \
  --optimizer adamw \
  --lr 3e-4 \
  --max-steps 2000 \
  --device cuda