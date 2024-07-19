import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mislabel.utils import run_experiment


experiments = ["iris", "newsgroups", "covtype", "mnist"]

all_results = []

for experiment in experiments:
    results = run_experiment(experiment, n_runs=20)
    results["dataset"] = experiment
    results.sort_values(by=["algorithm"], inplace=True)
    all_results.append(results)

    total_results = results.copy()
    total_results["method"] = "average"
    all_results.append(total_results)

results = pd.concat(all_results)

# Rename ModelScorer to ModelEnsemble, AUMScorer to AUM and RandomFrestScorer to RandomForest
results["algorithm"] = results["algorithm"].replace(
    {"ModelScorer": "ModelEnsemble", "AUMScorer": "AUM", "RandomForestScorer": "RandomForest"}
)

# Rename columns for better readability in the plot
results = results.rename(
    columns={
        "algorithm": "Algorithm",
        "fraction": "Mislabel fraction",
        "auc_score": "AUC score",
        "method": "Noise",
        "dataset": "Dataset",
    }
)

def output_appendix(results, file="results/appendix_results.tex"):

    # Since we have multiple runs, calculate the average of the runs with their standard deviation (use plus-minus symbol)
    latex = results.groupby(["Dataset", "Mislabel fraction", "Algorithm", "Noise"])["AUC score"].agg(
        ["mean", "std"]
    ).rename(columns={"mean": "AUC score", "std": "std"})

    # Output to latex
    with open(file, "w") as f:
        s = latex.to_latex(formatters={"name": str.upper}, float_format = "{:.3f}".format)
        s = s.replace("\\begin{tabular}", "\\begin{longtable}")
        s = s.replace("\\end{tabular}", "\\end{longtable}")
        f.write(s)


def output_graph(results, file="results/plots_datasets.pdf"):
    # Draw a dotted horizontal line at y=0.5
    #plt.axhline(y=0.5, color="black", linestyle="--")

    # Set the y axis from 0.4 to 1.0
    plt.ylim(0.45, 1.0)

    # Ensure that ModelScorer is always blue, AUMScorer is orange, and RandomForestScorer is green
    colors = {"ModelScorer": "blue", "AUMScorer": "orange", "RandomForestScorer": "green"}

    g = sns.FacetGrid(results, row="Dataset", col="Noise", margin_titles=True, height=2.5)
    g.map(sns.lineplot, "Mislabel fraction", "AUC score", "Algorithm", color=colors)

    # Add white space below plot
    plt.subplots_adjust(bottom=0.1)

    # Add legend to the bottom of the plot (instead of right
    plt.legend(loc="lower center", bbox_to_anchor=(-1, -0.5), ncol=3)

    # plt.show()
    plt.savefig(file)

output_appendix(results)
output_graph(results)

# Average over all Noise levels
latex2 = results.groupby(["Dataset", "Algorithm", "Noise"])["AUC score"].agg(
    ["mean", "std"]
).rename(columns={"mean": "AUC score", "std": "std"})

with open("results_table_summarized.tex", "w") as f:
    s = latex2.to_latex(formatters={"name": str.upper}, float_format = "{:.3f}".format)
    f.write(s)
exit(0)
# Create table with index noise (ascending), and then algorithm (ModelEnsemble, AUM, RandomForest) for each dataset,, with the score
#print(results.set_index(["Mislabel fraction", "Algorithm", "dataset", "noise_method"])["AUC score"].unstack())



