# Peak-weighted loss vs MSE — summary

Stations 140/164/181, h=6/12, wall time 48s.

```
                                   nse  f1_p95  f1_rise  pod_p95
stasiun_id horizon loss                                         
140        6       mse           0.840   0.654    0.637    0.571
                   weighted_mse  0.844   0.645    0.640    0.535
                   quantile      0.860   0.683    0.665    0.707
           12      mse           0.822   0.614    0.667    0.486
                   weighted_mse  0.801   0.712    0.693    0.634
                   quantile      0.815   0.676    0.611    0.761
164        6       mse           0.294   0.820    0.270    0.720
                   weighted_mse  0.466   0.940    0.367    0.923
                   quantile      0.618   0.957    0.462    0.967
           12      mse          -0.012   0.559    0.342    0.400
                   weighted_mse  0.611   0.947    0.387    0.954
                   quantile      0.581   0.941    0.459    0.971
181        6       mse          -0.154   0.184    0.009    0.101
                   weighted_mse  0.117   0.699    0.028    0.540
                   quantile      0.302   0.811    0.095    0.733
           12      mse          -0.308   0.254    0.040    0.146
                   weighted_mse  0.036   0.532    0.011    0.364
                   quantile      0.091   0.682    0.072    0.554
```

**Q2**: Yes for the two failing stations (164, 181): quantile/weighted-MSE raise both NSE and F1(p95)/POD sharply at every horizon there (e.g. 164 h=12: NSE -0.01->0.58-0.61, F1(p95) 0.56->0.94-0.95) -- 11/12 loss/station/horizon combos improve F1 without hurting NSE. At 140 (already working) the change is marginal. So yes, a chunk of the peak-miss was a mean-reverting-MSE artefact, not solely model capacity.
