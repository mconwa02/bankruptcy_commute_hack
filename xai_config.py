import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def shap_features_bar_plot(clf, x_train, colour):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_train)
    shap.summary_plot(shap_values[1], x_train, plot_type="bar", color=colour)


def plot_single_shap_values(clf, df, sample):
    """java script won't work in pycharm"""
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(df.iloc[sample, :])
    shap.initjs()
    return shap.force_plot(
        explainer.expected_value[1],
        shap_values[1],
        df.iloc[sample, :],
        matplotlib=False,
    )


def shap_violin_plot(clf, x_train):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_train)
    shap.summary_plot(shap_values[1], x_train, plot_type="violin")


def shap_values_importance_plot(clf, df, sample):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(df.iloc[sample, :])
    shap_df = pd.DataFrame(
        data=shap_values[1], index=df.columns, columns=["Shap Value"]
    ).sort_values(by=["Shap Value"], ascending=False)
    sample_df = pd.DataFrame(df.iloc[sample, :])
    shap_output = shap_df.join(sample_df, how="inner")
    temp_df = shap_output
    temp_df.sort_values(by=["Shap Value"], ascending=True, inplace=True)
    temp_df["positive"] = temp_df["Shap Value"] > 0
    index = np.arange(len(temp_df.index))
    plt.barh(
        index,
        temp_df["Shap Value"],
        color=temp_df.positive.map({True: "g", False: "c"}),
    )
    plt.yticks(index, temp_df.index)
    plt.title("Importance of shap features")
    plt.tight_layout()
    plt.show()
    rt_df = shap_output.sort_values(by=["Shap Value"], ascending=False)
    rt_df.columns = ["Shap Value", "Feature Value", "positive"]
    print(rt_df)
