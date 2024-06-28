import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mislabel.utils import run_experiment

experiment = "covtype"

# Check if file exists
results = run_experiment(experiment, n_runs=20)

# Filter out only runs using the nnar method
#results = results[results["method"] == "nnar"]

# Rename columns for better readability in the plot
results = results.rename(
    columns={
        "algorithm": "Algorithm",
        "fraction": "Fraction of mislabeled samples",
        "auc_score": "AUC score",
    }
)
# Draw a dotted horizontal line at y=0.5
plt.axhline(y=0.5, color="black", linestyle="--")
# Set title of scatter plot
plt.title(f"AUC scores for the {experiment} dataset")
sns.lineplot(data=results,
             x="Fraction of mislabeled samples", y="AUC score", hue="Algorithm")
plt.show()
