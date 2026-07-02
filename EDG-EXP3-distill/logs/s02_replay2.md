# Step 0.2 Replay 2:1 Weighted Dataset

Replay 2:1 uses all available unique replay rows, then repeats deterministically to reach 296 replay rows per arm.

## main

- core_rows: 148
- replay_rows: 296
- replay_unique_rows: 226
- replay_repeated_rows: 70
- total_rows: 444
- core_unique_episode_count: 74
- replay_unique_episode_count: 226
- source_distribution: {'replay_base_success': 296, 'teacher_t0': 74, 'teacher_t08': 74}
- partition_distribution: {'replay': 296, 'sampling_rescuable': 20, 'scaffold_only': 128}

## star

- core_rows: 148
- replay_rows: 296
- replay_unique_rows: 226
- replay_repeated_rows: 70
- total_rows: 444
- core_unique_episode_count: 31
- replay_unique_episode_count: 226
- source_distribution: {'replay_base_success': 296, 'star_pass16': 148}
- partition_distribution: {'replay': 296, 'sampling_rescuable': 148}

- leakage_asserts: {'main_intersect_heldout': 0, 'main_intersect_r_success_eval': 0, 'main_intersect_d_val': 0, 'star_intersect_heldout': 0, 'star_intersect_r_success_eval': 0, 'star_intersect_d_val': 0}
