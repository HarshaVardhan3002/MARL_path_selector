# Phase 0 Calibration Report

Config hash: `sha256:14b3827b380f3428c6f0e9ed491f7f06477717eb89ebd44f61dda632e4a200d2`

**Overall:** PASS

## Criteria
### path_lifetime_ks -- PASS
- ks_distance: 0.0596
- tolerance: 0.1500
- synthetic_median_sec: 27110.3012
- empirical_median_sec: 27000.0000

### rtt_marginal -- PASS
- synthetic_median_ms: 396.6178
- synthetic_iqr_ms: 153.1661
- empirical_median_p25_ms: 285.1635
- empirical_median_p75_ms: 581.9560
- empirical_iqr_median_ms: 296.7925

### bandwidth_marginal_tier50 -- PASS
- synthetic_median_mbps: 48.3449
- synthetic_iqr_mbps: 1.7730
- empirical_median_p25_mbps: 47.5212
- empirical_median_p75_mbps: 49.4750
- empirical_iqr_median_mbps: 1.7800

### path_discrepancy_rate -- PASS
- synthetic_rate: 0.3000
- empirical_mean_rate: 0.3073
- relative_gap: 0.0237
- tolerance: 0.2500

## Figures
- ![lifetime CDF](figures/lifetime_cdf.png)
- ![RTT overlay](figures/rtt_overlay.png)
- ![BW overlay](figures/bandwidth_overlay.png)