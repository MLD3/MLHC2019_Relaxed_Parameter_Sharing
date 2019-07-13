## Overview
- Requires: the raw csv files of the [MIMIC-III database](https://mimic.physionet.org/about/mimic/)
- Extract labels for ARF and shock

## Steps to reproduce results

0. Modify `config.yaml` to specify `mimic3_path` and `data_path`.

1. Data Extraction
    - Execute `python -c "from extract_data import *; check_nrows();"` to verify the integrity of raw csv files.
    - Run `python extract_data.py`.

1. Labels & Cohort definitions
    - Run `python generate_labels.py` to generate the event onset time and labels for two outcomes: ARF and shock. The output should be
        ```
        ARF:    {0: 13125, 1: 10495}    N = 23620
        Shock:  {0: 16629, 1: 6991}     N = 23620
        ```
