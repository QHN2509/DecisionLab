# choices13k data source

DecisionLab uses the original public choices13k repository:

- Repository: <https://github.com/jcpeterson/choices13k>
- Pinned revision: `821ae7e88386b508ebb46fae76fac63cb62ec876`
- Files: `c13k_selections.csv` and `c13k_problems.json`
- Checksums and immutable download URLs: `configs/choices13k_manifest.json`

The raw files are downloaded byte-for-byte, checksum-verified, made read-only, ignored by Git, and never transformed in place. The JSON keys are zero-based CSV row indexes; they are not `Problem` IDs.

## Target

`bRate` is the mean of participant-level Gamble B selection rates for a problem-condition row. Each participant completed five trials for a problem. Therefore, `bRate` is an aggregate continuous response in `[0, 1]`, not an individual binary choice. `n` is the number of participants contributing to the row and is measurement metadata rather than a default feature.

The CSV contains 14,568 condition rows but 13,006 problem IDs. Some problems have both no-feedback and feedback rows. These rows must remain grouped in later train/test splitting.

## Citation

Peterson, J. C., Bourgin, D. D., Agrawal, M., Reichman, D., & Griffiths, T. L. (2021). Using large-scale experiments and machine learning to discover theories of human decision-making. *Science, 372*(6547), 1209–1214. <https://doi.org/10.1126/science.abe2629>

Also cite the dataset repository as requested by its maintainers. See the upstream README for its additional Bourgin et al. (2019) citation.

