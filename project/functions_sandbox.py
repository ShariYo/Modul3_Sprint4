import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import precision_score, recall_score
from sklearn.impute import SimpleImputer


def cleaner(df, info=True):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.rename(columns=lambda x: x.lower(), inplace=True)
    df_duplicates = df.duplicated().any()
    df_nan = df.isna().any().any()
    df_empty = (df == "").any().any()

    if info:
        print("All columns empty spaces have been stripped.")
        print("All columns names have been converted to lowercase.\n")
        print(f"Is there any duplicates?: {df_duplicates}")
        print(f"Is there any NaN numbers?: {df_nan}")
        print(f"Is there any empty cells?: {df_empty}")

    return df


def column_delete(df, nan_perc=40):
    """
    Removes columns from a DataFrame that have NaN values exceeding a specified percentage.

    Parameters:
        df (pd.DataFrame): The DataFrame from which to delete columns.
        nan_perc (float): The threshold percentage of NaN values for column deletion (default is 40%).

    Returns:
        pd.DataFrame: A DataFrame with columns removed where the percentage of NaN values exceeds `nan_perc`.

    Prints:
        List of column names dropped due to high NaN percentage.
    """

    to_del = []
    for col in df.columns:
        percentage = df[col].isna().mean() * 100
        if percentage > nan_perc:
            to_del.append(col)
    df = df.drop(columns=to_del)
    print(f"Columns with > {nan_perc}% NaN values dropped: \n{to_del}")

    return df, to_del


def feature_log(df, features):
    for feature in features:
        if (df[feature] < -1).any():
            print(
                f"Feature: {feature} has negative numbers and cannot be processed logarithmically!"
            )
            pass
        else:
            df[feature] = np.log1p(df[feature])

    return df


def calc_vif(x):
    vif = pd.DataFrame()
    vif["variables"] = x.columns
    vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif = vif.sort_values(by="VIF", ascending=False)

    return vif


def nan_imput(df, numericals, categoricals):
    """
    Imputes missing values in a DataFrame for specified numerical and categorical columns.

    Parameters:
        df (pd.DataFrame): The DataFrame containing missing values to impute.
        numericals (list of str): List of column names with numerical data to impute using the median.
        categoricals (list of str): List of column names with categorical data to impute using the most frequent value.

    Returns:
        pd.DataFrame: The DataFrame with missing values imputed in specified numerical and categorical columns.
    """
    # create imputer variables for numeric and categoric features
    numerical_imp = SimpleImputer(strategy="median")
    categorical_imp = SimpleImputer(strategy="most_frequent")

    # fill empty cells using imputation method
    df[numericals] = numerical_imp.fit_transform(df[numericals])
    df[categoricals] = categorical_imp.fit_transform(df[categoricals])

    return df


def remove_outliers_iqr(df, column):
    # Check if column is a list
    if isinstance(column, str):
        column = [column]

    initial_row_count = df.shape[0]

    # remove outliers
    clean = df.copy()
    for col in column:
        Q1 = clean[col].quantile(0.25)
        Q3 = clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = clean[(clean[col] < lower_bound) | (clean[col] > upper_bound)]
        clean = clean[(clean[col] >= lower_bound) & (clean[col] <= upper_bound)]

    final_row_count = clean.shape[0]
    percent_removed = ((initial_row_count - final_row_count) / initial_row_count) * 100
    print(f"Percentage of data removed: {percent_removed:.2}%")

    return clean


def model_result_calc(target_test, target_predicted, pos_label):
    precision = precision_score(target_test, target_predicted, pos_label=pos_label)
    recall = recall_score(target_test, target_predicted, pos_label=pos_label)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    print(f"Precision score: {precision:.2f}")
    print(f"Recall score: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")

    pass


def reg_formula(model, X):
    coefficients = model.params
    formula = f"y = {coefficients.iloc[0]:.4f}"
    for i in range(1, len(coefficients)):
        formula += f" + {coefficients.iloc[i]:.4f}*{X.columns[i]}"

    return formula


def f_histogram(
    xaxis,
    bins=20,
    kde=False,
    figsize=(6, 4),
    label=None,
    xlabel=None,
    title=None,
    hue=None,
    palette=None,
    alpha=None,
    log=False,
):
    plt.figure(figsize=figsize)
    sns.histplot(
        x=xaxis,
        bins=bins,
        label=label,
        kde=kde,
        hue=hue,
        palette=palette,
        alpha=alpha,
        log_scale=log,
    )
    plt.xlabel(xlabel)
    plt.title(title, size=14, fontweight="bold", ha="center")

    return


def f_barplot(xaxis, figsize=(6, 4), xlabel=None, title=None):

    plt.figure(figsize=figsize)
    sns.set_palette("crest")
    ax = xaxis.plot(kind="bar", width=0.8)
    for container in ax.containers:
        ax.bar_label(container)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(True)
    ax.set_frame_on(False)
    plt.tight_layout()
    plt.xticks(rotation=0)
    plt.xlabel(xlabel)
    plt.title(title, size=14, fontweight="bold", ha="center")
    plt.legend()

    return


def f_boxplot(
    data=None,
    xaxis=None,
    yaxis=None,
    hue=None,
    figsize=(5, 3),
    showfliers=False,
    ylabel=None,
    title=None,
):
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=data,
        x=xaxis,
        y=yaxis,
        hue=hue,
        showfliers=showfliers,
        flierprops=dict(markerfacecolor="red", marker="o"),
        width=0.9,
        palette="deep",
    )
    plt.ylabel(ylabel)
    plt.title(title, size=14, fontweight="bold", ha="center")
    plt.tight_layout()
    plt.legend()

    return


def f_countplot(
    data=None,
    xaxis=None,
    yaxis=None,
    hue=None,
    stat="count",
    title=None,
    xlabel=None,
    ylabel=None,
    order=None,
    palette=None,
    figsize=(5, 3),
):
    plt.figure(figsize=figsize)
    sns.set_palette("crest")
    ax = sns.countplot(
        data=data, x=xaxis, y=yaxis, hue=hue, stat=stat, palette=palette, order=order
    )
    for container in ax.containers:
        ax.bar_label(container)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(True)
    ax.set_frame_on(False)
    plt.title(title, size=14, fontweight="bold", ha="center")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    return


def f_displot(
    data=None,
    xaxis=None,
    yaxis=None,
    hue=None,
    multiple="layer",
    palette=None,
    kind="hist",
    kde=None,
    bins=30,
    title=None,
    xlabel=None,
    ylabel=None,
    figsize=(5, 3),
):
    sns.displot(
        data=data,
        x=xaxis,
        hue=hue,
        multiple=multiple,
        palette=palette,
        kind=kind,
        kde=kde,
        bins=bins,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, size=14, fontweight="bold", ha="center")
    plt.tight_layout()

    return


def f_heatmap(data, figsize=(6, 4), title=None):
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(data.corr()))
    sns.heatmap(data, annot=True, cmap="Greens", mask=mask, center=0)
    plt.title(title, size=14, fontweight="bold", ha="center")

    return
