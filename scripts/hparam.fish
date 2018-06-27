for val in (seq 12)
  python scripts/train_infovae.py \
  --batch_size=50 \
  --logdir=/tmp/infovae/$val \
  --n_hidden=$val \
  --epochs=100 \
  --width=8
end
