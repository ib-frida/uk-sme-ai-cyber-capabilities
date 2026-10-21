# Organisational Capability Configurations in UK SMEs

**Implications for secure AI adoption and cyber resilience**

Analysis code for my MSc Data Science and its Applications dissertation (University of Essex, School of Mathematics, Statistics and Actuarial Science, August 2026), carried out as a placement with the Institute for Analytics and Data Science (IADS).

## The question

UK SMEs are being asked to adopt AI and strengthen their cyber defences at the same time. Using two national statistical releases, this project asks how AI capability and cybersecurity capability are configured inside UK SMEs, and what those configurations imply for secure AI adoption.

## Key findings

- **AI-specific governance lags general cyber capability.** Among firms using or considering AI, the share with AI-specific cyber practices sits 13 to 15 percentage points below their general cyber capability from the small size band upwards, and 78.5% of AI-engaged micro firms report no such practices.
- **Sector adoption varies widely,** from 35.2% (Defence) to 51.2% (Life Sciences).
- **Investment is asymmetric, not absent.** Smaller firms hold technical controls at roughly two-thirds to four-fifths of the large-firm rate, but governance, response, assurance and training capabilities at roughly a fifth to a half.

## Methods

Both sources publish population-weighted aggregate rates (four size bands, six sectors), with no record-level data. The code is written so that no firm-level estimate can be produced by accident.

- **Interval estimation:** Wald, Wilson, Clopper-Pearson, Agresti-Coull and Jeffreys intervals compared by Monte Carlo simulation (20,000 replications per cell); Wilson adopted.
- **Comparisons:** two-proportion z-tests, Newcombe hybrid-score intervals and Cohen's h for 30 pairwise size-band contrasts, with Holm-Bonferroni family-wise error control.
- **Association:** Spearman rank correlation with exact permutation p-values (n = 4).
- **Exploratory partitioning:** k-means on standardised band-level features, k chosen by silhouette.
- **Robustness:** six composite-weighting schemes, leave-one-feature-out partitions and 50 random seeds.
- **Reproducibility:** label-based extraction from the source spreadsheets, a fixed seed (42), and an automated verification pass that recomputes every value quoted in the dissertation.

## Repository layout

```
01_bics_csbs_descriptive.ipynb   exploratory notebook
analysis/
  load_data.py                   loading, cleaning and indicator extraction
  stats_core.py                  interval estimators, tests, effect sizes
  model_selection.py             Monte Carlo and multiplicity comparisons
  run_analysis.py                full analysis for the three research questions
  make_figures.py                figures for the results chapter
data/                            download instructions for the source spreadsheets
```

## Running it

Requires Python 3.11.

```bash
# first download the two spreadsheets into data/ (see data/README.md)
pip install -r requirements.txt
cd analysis
python run_analysis.py      # tables, results.json and verification report in ../out
python make_figures.py      # figures in ../figures
```

## Data

- **ONS Business Insights and Conditions Survey (BICS), AI adoption tables, Wave 147.**
- **DSIT Cyber Security Breaches Survey 2025/26.**

Both are UK official statistics under the Open Government Licence v3.0. The spreadsheets are not redistributed here; [data/README.md](data/README.md) links to the original releases.

## Author

Frida Ibarra · [LinkedIn](https://www.linkedin.com/in/fridaibarra/)

Supervisors: Dr Daniel Ahelegbey, Prof Haris Mouratidis, Dr Marta Fernández de Arroyabe Arranz.

## Licence

Code released under the [MIT Licence](LICENSE). Data remain subject to the Open Government Licence v3.0.
