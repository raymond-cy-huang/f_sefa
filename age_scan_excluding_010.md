# Age-direction scan excluding eigen_010

Based on the regenerated `eigen_000–199` contact sheets, the strongest
secondary age-like directions appear to be:

| Rank | Direction | Visual age effect | Main entanglement |
|---|---|---|---|
| 1 | `eigen_004` | Positive direction consistently moves the three identities toward babies/younger faces; negative direction toward adults | Some gender/identity change, but age effect is dominant |
| 2 | `eigen_001` | Clear adult/older-face shift at positive strength for multiple identities | Strong gender and hairstyle coupling |
| 3 | `eigen_003` | Positive direction often moves toward older/adult faces | Gender, hair and lighting coupling |
| 4 | `eigen_005` | Weaker young-to-adult tendency in some identities | Identity and gender leakage; less consistent than `004` |

## Interpretation

`eigen_004` is the best candidate for the second-strongest age-related
direction outside `eigen_010`. It is not a clean age-only direction, but its
positive/negative changes are more consistently age-like than the other
non-`010` candidates.

`eigen_001` and `eigen_003` should be treated as entangled age/gender
directions, not independent age controls. `eigen_005` is tentative.

This is a visual ranking from fixed-strength contact sheets, not a supervised
age-regressor score. A next step would be to run an age estimator on the
original and edited images and report `age_delta` separately from gender,
identity, and glasses leakage.
