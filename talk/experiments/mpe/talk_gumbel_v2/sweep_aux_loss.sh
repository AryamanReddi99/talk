for aux_coef in 0.001 0.01 0.1 0.2 0.5 0.8 1.0; do
  echo "=== aux_coef=${aux_coef} ==="
  python mappo_talk_v2.py \
    "aux_coef=${aux_coef}" \
    "codebook_size=1" \
    "sight_range=0.5" \
    "talk_config=_1_code_${aux_coef}_h_aux_0.5_sr"
done
echo "Sweep complete."