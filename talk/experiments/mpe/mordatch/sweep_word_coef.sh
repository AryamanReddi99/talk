for word_coef in 0.0 0.001 0.01 0.05 0.1; do
  echo "=== word_coef=${word_coef} ==="
  python mappo_mordatch.py \
    "word_coef=${word_coef}" \
    "sight_range=1.5" \
    "talk_config=_1.5_sr_${word_coef}_word"
done
echo "Sweep complete."
