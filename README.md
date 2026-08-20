# Toyota Corolla Exploratory Data Analysis

Exploratory data analysis of 1,436 Toyota Corolla listings. The project addresses the following question:

> Do the data indicate distinct groups of Toyota Corolla vehicles based on age, mileage, engine characteristics, weight, fuel type, and price? If so, how do the groups differ?

The assignment permits visualizations, descriptive statistics, and statistical tests, but does not permit clustering or predictive modeling.

## Main findings

- Vehicle age has the strongest association with price (Spearman rho = -0.842), followed by mileage (rho = -0.616).
- Fuel type alone does not separate prices well (Kruskal-Wallis p = 0.135, epsilon-squared = 0.003).
- A frequency table of fuel type, engine volume, and horsepower reveals seven interpretable technical segments without using a clustering algorithm.
- Technical segments separate prices much more strongly (epsilon-squared = 0.260), and the effect remains large within the 45-70 month age range (epsilon-squared = 0.170).
- The Petrol-Diesel price difference changes direction across age bands, so the overall fuel comparison is masked by vehicle age.

## Repository contents

- `Toyota_EDA_Question5.ipynb` - complete notebook with explanations and analysis code.
- `toyota_eda_question5.py` - reproducible command-line version of the analysis.
- `Toyota_Corolla_cars.xlsx` - course dataset used by the analysis.
- `presentation/Toyota_Corolla_Q5_HE.pptx` - Hebrew presentation.
- `outputs/figures/` - nine figures used in the analysis and presentation.
- `requirements.txt` - Python dependencies.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python toyota_eda_question5.py
```

The script writes regenerated figures, result tables, and a JSON summary under `outputs/`.

## Run in Google Colab

Upload `Toyota_EDA_Question5.ipynb` and `Toyota_Corolla_cars.xlsx` to the same Colab session, then run all cells from top to bottom.

## Methods

The analysis uses median and interquartile summaries, Spearman rank correlations, Kruskal-Wallis tests, Mann-Whitney comparisons, Holm correction for multiple testing, sensitivity checks for suspicious values, frequency tables, and stratified comparisons by vehicle age.

No clustering algorithm or predictive model is used.
