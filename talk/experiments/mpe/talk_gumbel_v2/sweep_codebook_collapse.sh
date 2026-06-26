# we are testing for codebooks collapse - both in terms of the codes within a single codebook,
# and codebooks across different hidden states

for sight_range in 0.25 0.5 1.5 -1.0; do
  for codebook_size in 4 8 16; do
    for aux_coef in 0.0; do
      echo "=== sight_range=${sight_range}, codebook_size=${codebook_size}, aux_coef=${aux_coef} ==="
      python mappo_talk_v2.py \
        "codebook_size=${codebook_size}" \
        "aux_coef=${aux_coef}" \
        "sight_range=${sight_range}" \
        "talk_config=_codebook_collapse"
    done
  done
done
echo "Sweep complete."