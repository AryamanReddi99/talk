for aux_coef in 0.005 0.008 0.012 0.015; do
  echo "=== aux_coef=${aux_coef} ==="
  python mappo_talk_v2.py \
    "aux_coef=${aux_coef}" \
    "codebook_size=8" \
    "sight_range=0.5" \
    "talk_config=_8_code_${aux_coef}_h_aux_0.5_sr"
done
echo "Sweep complete."