# Four-arm terminal: length and direction review

DEV only: 1,998 identical rows, terminal 3,426 updates. Recomputed from saved predictions and verified against existing collector to 1e-12. No new GPU evaluation or training.

Direction comparisons use the same GT interval length > 0.05m mask (11,273 intervals) for every model. Masked length below uses that identical mask. All-row length includes stationary intervals. Heading is displacement direction, not vehicle body yaw. These auxiliary averages are not a decomposition of PREFIX.

|Model|All-row length MAE (cm)|Masked length MAE (cm)|Masked heading MAE (deg)|Interval vector MAE (cm)|DEV PREFIX|
|---|---:|---:|---:|---:|---:|
|PARENT|7.547810|7.839868|0.719485|10.267029|0.151178860|
|P-CTRL|7.435309|7.692238|0.703365|10.124298|0.150285502|
|P-VECTOR|7.466049|7.727136|0.691055|10.084604|0.150329892|
|P-FINE|7.439559|7.690328|0.700604|10.125380|0.150370043|

P-VECTOR: compared with parent, both length and heading improve; compared with matched CTRL, heading improves but length worsens. Last 1-second interval vector error improves (15.814291 -> 15.693177cm), without an overall PREFIX gain.
P-FINE: compared with parent, both improve. On the common moving-interval mask, matched-CTRL length/heading both improve extremely slightly (7.692238 -> 7.690328cm; 0.703365 -> 0.700604deg); all-row length is slightly worse (7.435309 -> 7.439559cm), and total PREFIX is worse.

The matched CTRL comparison distinguishes the effect of additional training from the new modification. No statistical significance claim for component deltas, no server-score conversion, and no general rejection of longer/full-budget FINE training.

Prediction paths, hashes, per-interval results, paired descriptive sign changes and PREFIX session bootstrap intervals are in TERMINAL_LENGTH_HEADING_REVIEW.json.
