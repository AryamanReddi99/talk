for obs_pred_coef in 0.0 0.1 0.5 1.0 2.0; do
  echo "=== obs_pred_coef=${obs_pred_coef} ==="
  python mappo_mordatch.py \
    "obs_pred_coef=${obs_pred_coef}" \
    "sight_range=1.5" \
    "talk_config=_1.5_sr_${obs_pred_coef}_obspred"
done
echo "Sweep complete."
