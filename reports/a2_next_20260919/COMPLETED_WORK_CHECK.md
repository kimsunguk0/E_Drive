# Latest work check / 2026-09-19

Source base: be11e6f53b691bea38d2b470756783ac44424acc; tracked tree clean at entry. New files are isolated under a2_next_20260919.

|Arm|Status|Step|PREFIX|Role|
|---|---|---:|---:|---|
|BASE|completed|20554|0.165510648966|DEV|
|MH4|completed|20554|0.164495430150|DEV|
|QREFINE|completed|20554|0.164251769704|DEV|
|SIDE|completed|20554|0.172406018129|DEV|
|A2_FULL|completed|24931|0.088934085019|in-fit|

MR FULL is already submitted: server 0.18596892793122946. A2 FULL is completed, not server-scored.
No completed or running G0/G1 or learned-offset scene experiment was found in the current records.
Existing scene uses fixed calibration/pose projections, zero learned offsets. QREFINE changes reads, not locations.
G uses complete QREFINE terminal weights. Sampling independently uses the BASE upstream initializer and completed BASE control.
No FULL weights/teacher/cache enter DEV. No FULL restart. Physical GPUs 0–3 were idle on inspection; 4–7 are excluded.
Prior gradient diagnostic: 32 rows/8 batches on BASE. This user-specified extension: 32 effective batches/512 rows on QREFINE, separate shared parameter groups.
