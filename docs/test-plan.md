# Test Plan

## Unit tests included

- Original upload hash remains based on exact bytes.
- Resampling metadata distinguishes source rate from analysis rate.
- Empty and unsupported uploads are rejected safely.
- Very short signals return `inconclusive` instead of an unsupported result.
- Edit-screening scores are labelled as heuristic, not probability.

## Manual validation matrix

| Sample | Desired check | Important caveat |
|---|---|---|
| Natural speech with pauses | Avoid marking every pause as manipulation | A true edit can also fall in a quiet region |
| Controlled abrupt splice | Does signal analysis flag its vicinity? | Not all splices create discontinuities |
| Generated English speech | Test optional classifier output | Model's published benchmark may not generalise |
| Generated Tamil/Hindi speech | Measure cross-lingual generalisation | An output alone is not validation |
| Compressed genuine audio | Inspect false positives | Codec artifacts can trigger signal checks |
| Recording with music/noise | Examine robust behavior | Background noise can dominate speech features |

For evaluation, report confusion matrix, precision, recall, false-positive rate, false-negative rate and segment localisation tolerance where ground-truth boundaries exist. Do not describe model softmax outputs as calibrated probabilities without calibration testing.
