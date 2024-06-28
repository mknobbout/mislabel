import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mislabel.utils import run_experiment

# Check if file exists
run_experiment("newsgroups", n_runs=20)

results = pd.read_csv("results/iris.csv")
# Filter out only runs using the nnar method
# results = results[results["method"] == "nar"]

sns.lineplot(data=results, x="fraction", y="auc_score", hue="algorithm")
plt.show()
