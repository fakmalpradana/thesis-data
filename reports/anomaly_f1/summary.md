# Anomaly-based event F1 — summary

Median F1 per horizon x definition, LSTM vs persistence:

```
method               lstm  persistence
horizon definition                    
1       p95         0.904        0.911
        rise        0.870        0.783
        siaga3      0.791        0.889
6       p95         0.550        0.657
        rise        0.329        0.197
        siaga3      0.153        0.516
12      p95         0.342        0.497
        rise        0.172        0.131
        siaga3      0.000        0.267
24      p95         0.461        0.745
        rise        0.192        0.409
        siaga3      0.001        0.728
```
Per-station F1, h=6, `rise` definition:

```
method       lstm  persistence
stasiun_id                    
107         0.388        0.213
126         0.145        0.328
140         0.637        0.077
150         0.367        0.513
162         0.667        0.021
164         0.270        0.444
166         0.292        0.367
167         0.385        0.404
169         0.434        0.180
170         0.370        0.359
179         0.194        0.162
181         0.009        0.134
184         0.107        0.032
187         0.215        0.059
```

**Q1**: Switching from the datum-inconsistent Siaga-3 threshold to per-station anomaly definitions (p95, rise) does not rescue the LSTM at longer horizons: LSTM median F1 is still 0.55 (p95) / 0.33 (rise) at h=6 vs persistence 0.66 / 0.20 — the model under-predicts peaks regardless of how the event is defined, so this is a model behaviour, not a thresholding artefact.
