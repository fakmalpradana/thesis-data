# Siaga-3 exceedance F1/CSI — summary

Median F1/CSI across the 14 stations, per horizon, LSTM vs persistence (test period, Siaga 3):

```
                        f1    csi
horizon method                   
1       lstm         0.791  0.658
        persistence  0.889  0.800
6       lstm         0.153  0.091
        persistence  0.516  0.350
12      lstm         0.000  0.000
        persistence  0.267  0.155
24      lstm         0.001  0.000
        persistence  0.728  0.572
```

All 14 stations exceed Siaga 3 at least once in the test period; F1 is defined everywhere.

## Events vs ground truth

- **E2**: TMA network saw exceedance at stations [107, 126, 140, 150, 162, 164, 166, 167, 169, 170, 179, 181, 184, 187]; BPBD kejadian (kecamatan-month, summed over stations' kecamatan) 49; 10 PetaBencana reports within 3 km across stations.

- **E6**: TMA network saw exceedance at stations [107, 140, 150, 162, 166, 169, 170]; BPBD kejadian (kecamatan-month, summed over stations' kecamatan) 50; 2 PetaBencana reports within 3 km across stations.

- **E7**: TMA network saw exceedance at stations [107, 140, 162, 164, 179, 181]; BPBD not yet published for this month; 0 PetaBencana reports within 3 km across stations.
