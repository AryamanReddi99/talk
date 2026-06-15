for sight_range_scale in 0.0 0.25 0.50 0.75 1.0; do
  echo "=== sight_range_scale=${sight_range_scale} ==="
  python mappo_gru.py \
    "sight_range_scale=${sight_range_scale}" \
    
done
echo "Sweep complete."