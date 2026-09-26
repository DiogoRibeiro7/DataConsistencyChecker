"""Plotting mixin for DataConsistencyChecker.

This module hosts plotting helper methods that were previously part of
``check_data_consistency.py``. Splitting them into a mixin keeps the main
class easier to navigate.
"""

from __future__ import annotations

import datetime
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from matplotlib.patches import Rectangle

from .checker_state import CheckerState
from .checker_utils import clean_x_tick_labels, is_notebook, print_text, replace_special_with_space


class PlotsMixin(CheckerState):
    """Mixin providing plotting utilities for :class:`DataConsistencyChecker`."""


    def check_data_quality_by_feature_pairs(self, max_features_shown=30):
        """
        An alternative to check_data_quality(). This runs similar (though fewer) tests on pairs of features and, for
        each test, presents a matrix indicating for what fraction of the rows a given relationship between the features
        holds true.

        max_features_shown: int
            Where there are many features, it can be infeasible to show a heatmap of all features. However, it may be
            useful to render a heatmap for the first features. max_features_shown specifies how many features, at most,
            will be included in the heatmaps rendered.
        """

        def handle_results(matrix):
            if matrix.sum().sum() == 0:
                print_text("No instances found")
                return
            d = pd.DataFrame(matrix, columns=self.numeric_cols[:max_features_shown],
                             index=self.numeric_cols[:max_features_shown])
            d = d.replace(0, np.nan)
            num_feats = min(len(self.numeric_cols), max_features_shown)
            fig, ax = plt.subplots(figsize=(num_feats, num_feats))
            s = sns.heatmap(d, annot=True, fmt='.4f', cmap="seismic", linewidths=1.1, linecolor='blue', cbar=False)
            for _, spine in s.spines.items():
                spine.set_visible(True)
            plt.show()

        def run_test(func):
            matrix = np.zeros((min(len(self.numeric_cols), max_features_shown),
                               min(len(self.numeric_cols), max_features_shown)))
            for col_name_1_idx, col_name_1 in enumerate(self.numeric_cols[:max_features_shown]):
                for col_name_2_idx, col_name_2 in enumerate(self.numeric_cols[:max_features_shown]):
                    df = func(col_name_1, col_name_2)
                    matrix[col_name_1_idx, col_name_2_idx] = len(df) / self.num_rows
            handle_results(matrix)

        print_text("Blank cells indicate counts of zero for the corresponding pair of features.")
        print_text("## Fraction of rows where both columns contain the same value")
        def test_same(col_name_1, col_name_2):
            return self.orig_df[self.orig_df[col_name_1] == self.orig_df[col_name_2]]
        run_test(test_same)

        print_text("## Fraction of rows where both columns contain null")
        def test_null(col_name_1, col_name_2):
            return self.orig_df[(self.orig_df[col_name_1].isna()) & (self.orig_df[col_name_2].isna())]
        run_test(test_null)

        print_text("## Fraction of rows where both columns contain zero")
        def test_zero(col_name_1, col_name_2):
            return self.orig_df[(self.orig_df[col_name_1] == 0) & (self.orig_df[col_name_2] == 0)]
        run_test(test_zero)

        print_text("## Fraction of rows where both columns contain positive values")
        def test_both_positive(col_name_1, col_name_2):
            return self.orig_df[(self.numeric_vals_filled[col_name_1] >= 0) & (self.numeric_vals_filled[col_name_2] >= 0)]
        run_test(test_both_positive)

        print_text("## Fraction of rows where both columns contain negative values")
        def test_both_negative(col_name_1, col_name_2):
            return self.orig_df[(self.numeric_vals_filled[col_name_1] < 0) & (self.numeric_vals_filled[col_name_2] < 0)]
        run_test(test_both_negative)

        print_text("## Fraction of rows where one columns is the negative of the other")
        def test_negative(col_name_1, col_name_2):
            return self.orig_df[self.numeric_vals_filled[col_name_1] == (-1)*self.numeric_vals_filled[col_name_2]]
        run_test(test_negative)

        print_text("## Fraction of rows where one columns is an even multiple of the other")
        def test_even_multiple(col_name_1, col_name_2):
            val_arr_1 = self.numeric_vals_filled[col_name_1]
            val_arr_2 = self.numeric_vals_filled[col_name_2]
            test_series = np.where(val_arr_2 != 0, val_arr_1 / val_arr_2, val_arr_1).tolist()
            test_series = [float(x).is_integer() for x in test_series]
            return self.orig_df.loc[test_series]
        run_test(test_even_multiple)

        print_text("## Fraction of rows where one column is larger (using absolute values) than the other")
        def test_larger(col_name_1, col_name_2):
            return self.orig_df[abs(self.numeric_vals_filled[col_name_1]) > abs(self.numeric_vals_filled[col_name_2])]
        run_test(test_larger)

        print_text("## Fraction of rows where one columns is ten or more times (using absolute values) larger than the other")
        def test_much_larger(col_name_1, col_name_2):
            return self.orig_df[abs(self.numeric_vals_filled[col_name_1]) > abs((10.0)*self.numeric_vals_filled[col_name_2])]
        run_test(test_much_larger)

    # Public methods to output the results of the analysis in various ways
    ##################################################################################################################

    def plot_final_scores_distribution_by_row(self):
        """
        Display a probability plot and histogram representing the distribution of final scores by row.
        """

        if self.test_results_df is None or self.test_results_df.empty:
            return

        final_scores_df = self.test_results_df.copy()
        final_scores_df = final_scores_df.sort_values('FINAL SCORE', ascending=True)
        final_scores_df['Rank'] = range(self.num_rows)
        plt.subplots(figsize=(5, 3))
        s = sns.scatterplot(data=final_scores_df, x='Rank', y='FINAL SCORE')
        s.set_title("Distribution of Scores by Row, ordered lowest to highest scores")
        plt.show()

        plt.subplots(figsize=(5, 3))
        s = sns.histplot(data=final_scores_df, x='FINAL SCORE')
        s.set_title("Distribution of Scores per Row")
        plt.show()

        if final_scores_df['FINAL SCORE'].nunique() > 2:
            plt.subplots(figsize=(5, 3))
            s = sns.histplot(data=final_scores_df[final_scores_df['FINAL SCORE'] > 0], x='FINAL SCORE')
            s.set_title("Distribution of Scores per Row (Excluding Scores of 0)")
            plt.show()

    def plot_final_scores_distribution_by_feature(self):
        """
        Display a bar plot representing the distribution of final scores by feature.
        """

        final_scores_series = pd.Series(self.test_results_by_column_np.sum(axis=0)).sort_values(ascending=True).values
        final_scores_df = pd.DataFrame({'Feature': self.orig_df.columns,
                                        'Total Scores': final_scores_series})
        plt.subplots(figsize=(5, len(final_scores_df)*0.25))
        s = sns.barplot(data=final_scores_df, orient='h', y='Feature', x='Total Scores')
        s.set_title("Distribution of Total Scores per Column")
        plt.show()

    def plot_final_scores_distribution_by_test(self):
        """
        Display a bar plot representing the distribution of final scores by test.
        """

        if len(self.exceptions_summary_df) == 0:
            print("No exceptions found")
            return

        scores_by_test = pd.DataFrame(self.exceptions_summary_df.groupby('Test ID')['Number of Exceptions'].sum().sort_values())
        scores_by_test['Test ID'] = scores_by_test.index

        plt.subplots(figsize=(5, len(scores_by_test) * 0.25))
        s = sns.barplot(data=scores_by_test, orient='h', y='Test ID', x='Number of Exceptions')
        s.set_title("Distribution of Scores per Test")
        plt.show()

    def quick_report(self):
        """
        A convenience method, which calls several other APIs, to give an overview of the results in a single API.
        """

        def display_api_results(df, title):
            print("\n\n\n")
            if is_notebook():
                display(Markdown(f'# {title}'))
                display(df)
            else:
                print(title + ":")
                print(df)

        def display_plot(func, title):
            print("\n\n\n")
            if is_notebook():
                display(Markdown(f'# {title}'))
            else:
                print(title + ":")
            func()

        display_api_results(self.get_patterns_list(), 'Patterns List (short list only)')
        display_api_results(self.summarize_patterns_by_test_and_feature(), 'Patterns by Test and Feature')
        display_api_results(self.get_exceptions_list(), 'Exceptions List')
        display_api_results(self.summarize_exceptions_by_test_and_feature(), 'Exceptions Summary by Test and Feature')
        display_api_results(self.summarize_exceptions_by_test(), 'Exceptions Summary by Test')
        display_api_results(self.summarize_patterns_and_exceptions(), 'Summary of Patterns and Exceptions (all tests)')
        display_plot(self.plot_final_scores_distribution_by_row, "Final Scores by Row of the Data")
        display_plot(self.plot_final_scores_distribution_by_feature, "Final Scores by Feature")
        display_plot(self.plot_final_scores_distribution_by_test, "Final Scores by Test")


    def plot_columns_vs_final_scores(self):
        """
        Used to determine if there are any relationships between column values and the final scores of the rows. This
        displays tables and plots presenting any relationships found.
        """

        def clear_last_plots():
            if n_rows == 1:
                for i in range(num_feats, 4):
                    ax[i].set_visible(False)
            else:
                last_col_used = num_feats % 4
                for i in range(last_col_used, 4):
                    ax[n_rows-1][i].set_visible(False)

        if self.exceptions_summary_df is None or len(self.exceptions_summary_df) == 0:
            print("No exceptions found.")
            return

        df = self.orig_df.copy()
        df['FINAL SCORE'] = self.test_results_df['FINAL SCORE']

        num_feats = len(self.numeric_cols) + len(self.date_cols)
        feats = self.numeric_cols + self.date_cols
        if num_feats > 50:
            print(
                f"There are {num_feats} numeric and date features. Displaying only the 50 with the greatest correlation with the final score"
            )
            corr = {
                col: abs(df[col].astype(float).corr(df['FINAL SCORE'])) for col in feats
            }
            feats = [k for k, _ in sorted(corr.items(), key=lambda x: x[1], reverse=True)[:50]]
            num_feats = len(feats)

        if num_feats > 0:
            n_rows = math.ceil(num_feats / 4)
            fig, ax = plt.subplots(nrows=n_rows, ncols=4, figsize=(14, 4 * n_rows))
            for feat_idx, col_name in enumerate(feats):
                cur_ax = ax[feat_idx] if n_rows == 1 else ax[feat_idx // 4][feat_idx % 4]
                s = sns.scatterplot(data=df, x=df[col_name], y=df['FINAL SCORE'], ax=cur_ax)
                s.set(xlabel=None)
                s.set_title(col_name)
            clear_last_plots()
            plt.suptitle("Relationship of features to Final Score (Numeric and Date features)")
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.show()

        num_feats = len(self.binary_cols) + len(self.string_cols)
        feats = self.binary_cols + self.string_cols
        if num_feats > 50:
            print(
                f"There are {num_feats} numeric and date features. Displaying only the 50 with the greatest correlation with the final score"
            )
            corr = {}
            for col in feats:
                codes = pd.Categorical(df[col]).codes
                corr[col] = abs(pd.Series(codes).corr(df['FINAL SCORE']))
            feats = [k for k, _ in sorted(corr.items(), key=lambda x: x[1], reverse=True)[:50]]
            num_feats = len(feats)

        if num_feats > 0:
            n_rows = math.ceil(num_feats / 4)
            fig, ax = plt.subplots(nrows=n_rows, ncols=4, figsize=(14, 4 * n_rows))
            for feat_idx, col_name in enumerate(feats):
                vc = self.orig_df[col_name].value_counts()
                cur_ax = ax[feat_idx] if n_rows == 1 else ax[feat_idx // 4][feat_idx % 4]
                if len(vc) > 10:
                    common_vals = []
                    for v_idx in vc.index:
                        if vc[v_idx] > (self.num_rows / 10):
                            common_vals.append(v_idx)
                    if len(common_vals) == 0:
                        continue
                    map_dict = {x: x for x in common_vals}
                    sub_df = df.copy()
                    sub_df[col_name] = df[col_name].map(map_dict)
                    sub_df[col_name] = sub_df[col_name].fillna("Other")
                    s = sns.boxplot(data=sub_df,  x=col_name, y='FINAL SCORE', ax=cur_ax)
                else:
                    s = sns.boxplot(data=df,  x=col_name, y='FINAL SCORE', ax=cur_ax)
                s.set_title(col_name)

            clear_last_plots()
            plt.suptitle("Relationship of features to Final Score (String and Binary features)")
            plt.tight_layout()
            # See https://stackoverflow.com/questions/8248467/tight-layout-doesnt-take-into-account-figure-suptitle
            plt.subplots_adjust(top=0.90)
            plt.show()


    def save_image(self, f):
        """Save the current matplotlib figure to disk and reference it in HTML.

        Args:
            f: File handle used for HTML export.
        """
        image_file_name = f"output_{self.image_output_num}.png"
        full_image_file_name = os.path.join(self.output_folder, image_file_name)
        plt.savefig(full_image_file_name)
        self.image_output_num += 1
        f.write(f"<img src={image_file_name}><br>")

    def show_image(self, f):
        """Display the current figure or save it when exporting HTML.

        Args:
            f: Optional file handle for HTML export.
        """
        if f:
            self.save_image(f)
            plt.close("all")
        else:
            plt.show()

    def __plot_distribution(self, test_id, col_name, show_exceptions, display_info, f):
        """Plot value distribution highlighting flagged entries."""
        fig, ax = plt.subplots(figsize=(5, 3))
        s = sns.histplot(data=self.orig_df, x=col_name, color='blue', bins=100, ax=ax)

        if show_exceptions:
            if test_id in ['VERY_LARGE', 'LATE_DATES']:
                ax.axvspan(xmin=self.orig_df[col_name].min(), xmax=display_info['upper_limit'], facecolor='blue', alpha=0.3)
                ax.axvspan(xmin=display_info['upper_limit'], xmax=self.orig_df[col_name].max(), facecolor='red',  alpha=0.3)
            if test_id in ['VERY_SMALL', 'EARLY_DATES']:
                ax.axvspan(xmin=self.orig_df[col_name].min(), xmax=display_info['lower_limit'], facecolor='red', alpha=0.3)
                ax.axvspan(xmin=display_info['lower_limit'], xmax=self.orig_df[col_name].max(), facecolor='blue',  alpha=0.3)
            if test_id in ['VERY_SMALL_ABS']:
                ax.axvspan(xmin=self.orig_df[col_name].min(), xmax=-display_info['lower_limit'], facecolor='blue', alpha=0.3)
                ax.axvspan(xmin=-display_info['lower_limit'], xmax=display_info['lower_limit'], facecolor='red',  alpha=0.3)
                ax.axvspan(xmin=display_info['lower_limit'], xmax=self.orig_df[col_name].max(), facecolor='blue', alpha=0.3)
            if test_id in ['LESS_THAN_ONE']:
                if self.orig_df[col_name].min() < -1:
                    ax.axvspan(xmin=self.orig_df[col_name].min(), xmax=-1, facecolor='red', alpha=0.3)
                ax.axvspan(xmin=max(-1, self.orig_df[col_name].min()), xmax=min(1, self.orig_df[col_name].max()),
                           facecolor='blue',  alpha=0.3)
                if self.orig_df[col_name].max() > 1:
                    ax.axvspan(xmin=1, xmax=self.orig_df[col_name].max(), facecolor='red', alpha=0.3)
            if test_id in ['GREATER_THAN_ONE']:
                if self.orig_df[col_name].min() < -1:
                    ax.axvspan(xmin=self.orig_df[col_name].min(), xmax=-1, facecolor='blue', alpha=0.3)
                ax.axvspan(xmin=max(-1, self.orig_df[col_name].min()), xmax=min(1, self.orig_df[col_name].max()),
                           facecolor='red',  alpha=0.3)
                if self.orig_df[col_name].max() > 1:
                    ax.axvspan(xmin=1, xmax=self.orig_df[col_name].max(), facecolor='blue', alpha=0.3)
            if test_id in ['POSITIVE']:
                ax.axvspan(xmin=self.orig_df[col_name].min(), xmax=0, facecolor='red', alpha=0.3)
                ax.axvspan(xmin=0, xmax=self.orig_df[col_name].max(), facecolor='blue',  alpha=0.3)
            if test_id in ['NEGATIVE']:
                ax.axvspan(xmin=self.orig_df[col_name].min(), xmax=0, facecolor='blue', alpha=0.3)
                ax.axvspan(xmin=0, xmax=self.orig_df[col_name].max(), facecolor='red',  alpha=0.3)

        # Ensure there are not too many tick labels to be readable
        clean_x_tick_labels(fig, 1, ax)

        if show_exceptions:
            s.set_title(f"Distribution of {col_name} (Flagged values in red)")
        else:
            s.set_title(f"Distribution of {col_name}")

        # Find the flagged values and identify them on the plot
        if show_exceptions:
            results_col_name = self.get_results_col_name(test_id, col_name)
            results_col = self.test_results_df[results_col_name]
            flagged_idxs = np.where(results_col)
            flagged_vals = self.orig_df.loc[flagged_idxs, col_name].values
            for v in flagged_vals:
                s.axvline(v, color='red')
        self.show_image(f)

    def __draw_scatter_plot(self, df, test_id, x_col, y_col, y_is_calculated, columns_set, show_expections, display_info, f):
        """Draw scatter plots for test results."""

        def draw_diagonal(ax):
            if show_diagonal:
                ax.plot(xlim, ylim)

        def draw_kde(ax):
            if test_id in ['RARE_COMBINATION']:
                # Use a kde plot as the background
                sns.kdeplot(
                    data=df,
                    x=x_col,
                    y=y_col,
                    fill=True,
                    ax=ax)

        def apply_gridlines(ax):
            # For RARE_COMBINATION, draw the grid lines to make it more clear why certain values were flagged
            if test_id in ['RARE_COMBINATION']:
                for v in display_info['bins_1']:
                    if v not in [-np.inf, np.inf]:
                        ax.axvline(v, color='green', linewidth=1, alpha=0.3)
                for v in display_info['bins_2']:
                    if v not in [-np.inf, np.inf]:
                        ax.axhline(v, color='green', linewidth=1, alpha=0.3)

        def get_xy_lim(df, y_is_calculated):
            xlim = None
            ylim = None

            if x_col in self.numeric_cols:
                xlim = (df[x_col].min(), df[x_col].max())
                rng = xlim[1] - xlim[0]
                xlim = (xlim[0] - (rng / 50.0), xlim[1] + (rng / 50.0))

            if (y_col in self.numeric_cols) or y_is_calculated:
                ylim = (df[y_col].min(), df[y_col].max())
                rng = ylim[1] - ylim[0]
                ylim = (ylim[0] - (rng / 50.0), ylim[1] + (rng / 50.0))

            # todo: set xlim & ylim equal if can!
            # todo: put back -- does force x & y to use the same scale, which makes comparing easier.
            # if (x_col in self.numeric_cols) and (y_col in self.numeric_cols):
            #     xylim = (min(df[x_col].astype(float).min(), df[y_col].astype(float).min()),
            #              max(df[x_col].astype(float).max(), df[y_col].astype(float).max()))

            return xlim, ylim

        show_diagonal = test_id in ['MEAN_OF_COLUMNS', 'SUM_OF_COLUMNS', 'MIN_OF_COLUMNS', 'MAX_OF_COLUMNS',
                                    'LARGER_SAME_RANGE', 'SIMILAR_TO_PRODUCT', 'SIMILAR_TO_RATIO']

        if not y_is_calculated:
            df = self.orig_df[[x_col, y_col]].copy()
            if x_col in self.numeric_vals_filled:
                df[x_col] = self.numeric_vals_filled[x_col]
            if y_col in self.numeric_vals_filled:
                df[y_col] = self.numeric_vals_filled[y_col]

        if not show_expections:
            xlim, ylim = get_xy_lim(df, y_is_calculated)
            fig, ax = plt.subplots(figsize=(5, 4))
            draw_kde(ax)
            s = sns.scatterplot(
                data=df,
                x=x_col,
                y=y_col,
                color='blue',
                alpha=0.2,
                label='Not Flagged',
                ax=ax
            )
            s.set_title(f'Distribution of "{x_col}" and "{y_col}"')
            s.legend().remove()
            s.set_xlim(xlim)
            s.set_ylim(ylim)
            apply_gridlines(ax)
            draw_diagonal(ax)
            clean_x_tick_labels(fig, 1, ax)
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
            result_col_name = self.get_results_col_name(test_id, columns_set)
            df['Flagged'] = self.test_results_df[result_col_name]
            df_not_flagged = df[df['Flagged'] == 0]

            # Draw without the exceptions
            xlim, ylim = get_xy_lim(df_not_flagged, y_is_calculated)
            draw_kde(ax[0])
            s = sns.scatterplot(
                data=df_not_flagged,
                x=x_col,
                y=y_col,
                color='blue',
                alpha=0.2,
                label='Normal',
                ax=ax[0]
            )
            s.set_title(f'Distribution of \n"{x_col}" \nand \n"{y_col}" \n(excluding flagged values)')
            s.legend().remove()
            s.set_xlim(xlim)
            s.set_ylim(ylim)
            apply_gridlines(ax[0])
            draw_diagonal(ax[0])
            clean_x_tick_labels(fig, 2, ax[0])

            # Draw with and without the exceptions
            xlim, ylim = get_xy_lim(df, y_is_calculated)
            draw_kde(ax[1])
            s = sns.scatterplot(
                data=df[df['Flagged'] == 0],
                x=x_col,
                y=y_col,
                color='blue',
                alpha=0.2,
                label='Normal',
                ax=ax[1]
            )
            s = sns.scatterplot(
                data=df[df['Flagged'] == 1],
                x=x_col,
                y=y_col,
                color='red',
                alpha=1.0,
                label='Flagged',
                ax=ax[1]
            )
            s.set_title(f'Distribution of \n"{x_col}" \nand \n"{y_col}" \n(Flagged values in red)')
            s.legend().remove()
            apply_gridlines(ax[1])
            s.set_xlim(xlim)
            s.set_ylim(ylim)
            clean_x_tick_labels(fig, 2, ax[1])

        plt.tight_layout()
        self.show_image(f)

    def __plot_count_plot(self, column_name, f):
        """Visualize count of unique values."""
        fig, ax = plt.subplots(figsize=(4, 4))
        if column_name in self.date_cols:
            s = sns.countplot(orient='h', y=self.orig_df[column_name].fillna("NONE"))
        else:
            s = sns.countplot(orient='h', y=self.orig_df[column_name].fillna("NONE").str.strip())
        s.set_title(f"Counts of unique values in {column_name}")
        clean_x_tick_labels(fig, 1, ax)
        self.show_image(f)

    def __plot_heatmap(self, test_id, cols, f):  # noqa: ARG002
        """Display heatmap for counts of value combinations."""
        col_name_1, col_name_2 = cols
        plt.subplots(figsize=(4, 4))

        # todo: we may wish to include null as well, but need special handling below
        vals1 = self.orig_df[col_name_1].dropna().unique()
        vals2 = self.orig_df[col_name_2].dropna().unique()
        counts_arr = []
        for v1 in vals1:
            row_arr = []
            for v2 in vals2:
                row_arr.append(len(self.orig_df[(self.orig_df[col_name_1] == v1) & (self.orig_df[col_name_2] == v2)]))
            counts_arr.append(row_arr)
        df = pd.DataFrame(counts_arr, index=vals1, columns=vals2)
        s = sns.heatmap(df, cmap='Blues', linewidths=1.1, linecolor='black', annot=True, fmt="d")
        s.set_title(f"Counts of unique values in {col_name_1} and {col_name_2}")
        s.set_xlabel(col_name_2)
        s.set_ylabel(col_name_1)
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        self.show_image(f)

    def __draw_row_rank_plot(self, test_id, col_name, show_exceptions, f):
        """Plot column values against row index."""
        if not show_exceptions:
            fig, ax = plt.subplots(figsize=(8, 4))
            vals = self.orig_df[col_name]
            if col_name in self.date_cols:
                vals = [pd.to_datetime(x) for x in self.orig_df[col_name]]
            s = sns.scatterplot(x=self.orig_df.index, y=vals, color='blue', alpha=0.2, label='Not Flagged')
            if show_exceptions:
                s.set_title(f'Distribution of "{col_name}" (Flagged values in red)')
            else:
                s.set_title(f'Distribution of "{col_name}"')
            s.set_xlabel("Row Number")
            clean_x_tick_labels(fig, 1, ax)
            plt.legend().remove()
            self.show_image(f)
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(10, 4))

            # Find the flagged values and identify them on the plot
            results_col_name = self.get_results_col_name(test_id, col_name)
            results_col = self.test_results_df[results_col_name]
            not_flagged_idxs = np.where(~results_col)
            not_flagged_vals = self.orig_df.loc[not_flagged_idxs][col_name].values
            flagged_idxs = np.where(results_col)
            flagged_vals = self.orig_df.loc[flagged_idxs][col_name].values

            # Draw one plot without the flagged values
            s = sns.scatterplot(x=not_flagged_idxs[0], y=not_flagged_vals, color='blue', ax=ax[0])
            s.set_xlabel("Row Number")
            clean_x_tick_labels(fig, 2, ax[0])
            s.set_title("Values without exceptions")

            # Draw one plot with the flagged values
            s = sns.scatterplot(x=not_flagged_idxs[0], y=not_flagged_vals, color='blue', label="Not Flagged", ax=ax[1])
            s = sns.scatterplot(x=flagged_idxs[0], y=flagged_vals, color='red', label="Flagged", ax=ax[1])
            s.set_xlabel("Row Number")
            clean_x_tick_labels(fig, 2, ax[1])
            s.set_title("Values with exceptions")

            plt.legend().remove()
            plt.tight_layout()
            self.show_image(f)

    def __draw_box_plots(self, test_id, cols, columns_set, f):
        """Draw box plots to compare distributions."""
        # The first column is the string/binary value and the second is the numeric/date value
        # todo: this does not colour the outliers. We may wish to draw a histogram next to it, but this is somewhat
        # kludgy.
        fig, ax = plt.subplots(figsize=(8, 4))
        if cols[1] in self.numeric_cols:
            s = sns.boxplot(data=self.orig_df, orient='h', y=cols[0], x=cols[1])
        else:
            df = self.orig_df[cols].copy()
            df['Days Since Min Date'] = (pd.to_datetime(self.orig_df[cols[1]]) - pd.to_datetime(self.orig_df[cols[1]]).min()).dt.days
            s = sns.boxplot(data=df, orient='h', y=cols[0], x='Days Since Min Date')
        clean_x_tick_labels(fig, 1, ax)
        self.show_image(f)

        # Also draw a histogram of the relevant classes.
        results_col_name = self.get_results_col_name(test_id, columns_set)
        results_col = self.test_results_df[results_col_name]
        flagged_idxs = np.where(results_col)
        flagged_df = self.orig_df.loc[flagged_idxs]
        vals = flagged_df[cols[0]].unique()
        nvals = len(vals)
        fig, ax = plt.subplots(nrows=1, ncols=nvals, figsize=(nvals*4, 4))
        for v_idx, v in enumerate(vals):
            sub_df = self.orig_df[self.orig_df[cols[0]] == v]
            sub_flagged_df = flagged_df[flagged_df[cols[0]] == v]
            curr_ax = ax if nvals == 1 else ax[v_idx]
            s = sns.histplot(data=sub_df, x=cols[1], color='blue', bins=100, ax=curr_ax)
            flagged_vals = sub_flagged_df[cols[1]].values
            for fv in flagged_vals:
                # Add alpha, as in some cases the red lines are very close to the blue
                s.axvline(fv, color='red', alpha=0.5)
            s.set_title(f'Distribution of \n"{cols[1]}" where \n"{cols[0]}" is \n"{v}" \n(Flagged values in red)')
            clean_x_tick_labels(fig, nvals, curr_ax)
        plt.tight_layout()
        self.show_image(f)

    def __plot_larger_relationship(self, test_id, cols, columns_set, show_exceptions, display_info, f):  # noqa: ARG002
        """Plot relationships where one column is larger."""
        col_medians = [self.column_medians[c] for c in cols]
        cols = np.array(cols)[np.argsort(col_medians)]

        fig, ax = plt.subplots(figsize=(8, len(cols)))
        df_arr = []
        for c in cols:
            df = pd.DataFrame(self.numeric_vals[c])
            df.columns = ['Value']
            df['Feature'] = [c] * len(df)
            df_arr.append(df)
        df = pd.concat(df_arr)
        sns.boxplot(data=df.dropna(), orient='h', x='Value', y='Feature')
        if test_id in ['MUCH_LARGER']:
            plt.xscale('log')
        clean_x_tick_labels(fig, 1, ax)
        self.show_image(f)

    def __draw_network_plot(self, nodes, edges):
        """Visualize network of larger-than relationships."""
        """
        Each node represents one feature in the original data, and each edge represents a larger-than relationship.
        """

        def get_num_conflicts(node, y):
            num_conflicts = 0
            # Loop through each edge going back from node
            for edge in edges:
                if edge[0] != node[0]:
                    continue

                # Determine the x and y position of the other node in this edge
                other_node_feature = edge[1]
                x_pos_other_node, y_pos_other_node = positions[other_node_feature]

                # Loop through all nodes within the x range of this edge
                for n in nodes:
                    if (n[1] < x_pos_other_node) or (n[1] > node[1]):
                        continue
                    if (n[0] == other_node_feature) or (n[0] == node[0]):
                        continue

                    x_pos_middle_node, y_pos_middle_node = positions[n[0]]

                    # Determine the y value of the current edge, given the proposed y value for the node, at the x
                    # position of n
                    x_pos_middle_node = n[1]
                    y_diff = y - y_pos_other_node if y > y_pos_other_node else y_pos_other_node - y
                    x_diff = node[1] - x_pos_other_node
                    frac_along_x = (x_pos_middle_node - x_pos_other_node) / x_diff
                    y_edge_at_n = y_pos_other_node + (frac_along_x * y_diff)

                    # If the edge is too close to n, consider this a conflict
                    if abs(x_pos_middle_node - y_edge_at_n) < 5:
                        num_conflicts += 1

            print(node, y, num_conflicts)
            return num_conflicts

        # Remove redundant edges. We create an nxn matrix, representing the relationship of one column being larger
        # than other. Initially these are 1 if a>b, and 0 otherwise. We then replace as many 1 values with 2 (indicating
        # redundant) as we can. These are of the form a>c, where we also have a>b and b>c.
        mat = np.zeros((len(nodes), len(nodes)))
        for edge in edges:
            mat[nodes.index(edge[1]), nodes.index(edge[0])] = 1
        num_chanaged = 1
        print(nodes)
        print('before:')
        print(mat)
        while num_chanaged > 0:
            num_chanaged = 0
            for r in range(len(nodes)):
                for c in range(len(nodes)):
                    if mat[r][c] != 1:
                        continue
                    for other_r in range(len(nodes)):
                        if (mat[other_r][c] > 0) and (mat[r][other_r] > 0):
                            mat[r][c] = 2
                            num_chanaged += 1
        print('after:')
        print(mat)

        # Include the x position with each node and sort these left to right
        nodes = [(node, self.column_medians[node]) for node in nodes]
        nodes = sorted(nodes, key=lambda x: x[1])
        node_names = [n[0] for n in nodes]

        # Define the location of each node.
        positions = {}
        for node_idx, node in enumerate(nodes):
            y_pos = 50
            num_conflicts_arr = []
            if node_idx > 0:
                for y in range(5, 100, 5):
                    num_conflicts_arr.append(get_num_conflicts(node, y))
                idx_arr = np.argsort(num_conflicts_arr)
                # If multiple positions are equally unobstructed, favour the position closest to 50.
                y_pos = (idx_arr[0] + 1) * 5
            positions[node[0]] = (node[1], y_pos)

        plt.figure(figsize=(max(4, len(nodes) * 2), 4))

        # Draw the nodes
        for node in nodes:
            posn = positions[node[0]]
            plt.scatter(posn[0], posn[1], label=node, s=500, c='skyblue')
            plt.text(posn[0], posn[1]-10, node, fontsize=9, color='black', fontweight='bold', ha='center', va='center')

        # Draw the edges
        for edge in edges:
            node_0 = edge[0]
            node_1 = edge[1]
            if mat[node_names.index(node_1)][node_names.index(node_0)] == 1:
                p_1x = positions[node_0][0]
                p_1y = positions[node_0][1]
                p_2x = positions[node_1][0]
                p_2y = positions[node_1][1]
                plt.arrow(p_1x, p_1y, p_2x - p_1x, p_2y - p_1y, length_includes_head=True, head_width=3, head_length=3)

        plt.axis('off')
        plt.ylim(0, 100)
        plt.title("Relationships of Column Magnitudes")
        plt.show()
        print_text('Edges indicate pairs of columns where one column, row by row, contains strictly larger values '
                    'than the other column. Redundant edges are removed. The x-position of the nodes represents the '
                    'median value of the column.')

    def _draw_results_plots(self, test_id, cols, columns_set, show_exceptions, display_info, f):
        """Dispatch to appropriate plotting routine for a test."""

        if test_id in ['UNUSUAL_ORDER_MAGNITUDE', 'FEW_NEIGHBORS', 'FEW_WITHIN_RANGE', 'VERY_SMALL', 'VERY_LARGE',
                       'VERY_SMALL_ABS', 'LESS_THAN_ONE', 'GREATER_THAN_ONE', 'NON_ZERO', 'POSITIVE', 'NEGATIVE',
                       'EARLY_DATES', 'LATE_DATES']:
            self.__plot_distribution(test_id, cols[0], show_exceptions, display_info, f)

        if test_id in ['LARGER_DIFF_RANGE', 'LARGER_SAME_RANGE', 'MUCH_LARGER']:
            if len(cols) == 2:
                self.__draw_scatter_plot(
                    df=self.orig_df,
                    test_id=test_id,
                    x_col=cols[0],
                    y_col=cols[1],
                    y_is_calculated=False,
                    columns_set=columns_set,
                    show_expections=show_exceptions,
                    display_info=display_info,
                    f=f)
            self.__plot_larger_relationship(test_id, cols, columns_set, show_exceptions, display_info, f)

        if test_id in ['SIMILAR_WRT_RATIO', 'SIMILAR_WRT_DIFF', 'SIMILAR_TO_INVERSE', 'CORRELATED_DATES',
                       'SIMILAR_TO_NEGATIVE', 'CORRELATED_NUMERIC', 'RARE_COMBINATION', 'BINARY_MATCHES_VALUES']:
            self.__draw_scatter_plot(
                df=self.orig_df,
                test_id=test_id,
                x_col=cols[0],
                y_col=cols[1],
                y_is_calculated=False,
                columns_set=columns_set,
                show_expections=show_exceptions,
                display_info=display_info,
                f=f)

        if test_id in ['SAME_VALUES']:
            if self.orig_df[cols[0]].nunique() > 5:
                self.__draw_scatter_plot(
                    df=self.orig_df,
                    test_id=test_id,
                    x_col=cols[0],
                    y_col=cols[1],
                    y_is_calculated=False,
                    columns_set=columns_set,
                    show_expections=show_exceptions,
                    display_info=display_info,
                    f=f)
            else:
                self.__plot_heatmap(test_id, cols, f)

        if test_id in ['MEAN_OF_COLUMNS', 'SUM_OF_COLUMNS', 'MIN_OF_COLUMNS', 'MAX_OF_COLUMNS',
                       'CONSTANT_SUM', 'CONSTANT_DIFF', 'CONSTANT_PRODUCT', 'CONSTANT_RATIO']:
            df = self.orig_df.copy()
            if test_id in ['MEAN_OF_COLUMNS']:
                calculated_col = 'Mean'
                df[calculated_col] = display_info['Mean']
            if test_id in ['SUM_OF_COLUMNS']:
                calculated_col = 'Sum'
                df[calculated_col] = display_info['Sum']
            if test_id in ['MIN_OF_COLUMNS']:
                calculated_col = 'Min'
                df[calculated_col] = display_info['Min']
            if test_id in ['MAX_OF_COLUMNS']:
                calculated_col = 'Max'
                df[calculated_col] = display_info['Max']
            if test_id in ['CONSTANT_SUM']:
                calculated_col = 'SUM'
                df[calculated_col] = display_info['Sum']
            if test_id in ['CONSTANT_DIFF']:
                calculated_col = 'DIFF'
                df[calculated_col] = display_info['Diff']
            if test_id in ['CONSTANT_PRODUCT']:
                calculated_col = 'PRODUCT'
                df[calculated_col] = display_info['Product']
            if test_id in ['CONSTANT_RATIO']:
                calculated_col = 'RATIO'
                df[calculated_col] = display_info['Ratio']
            self.__draw_scatter_plot(df=df,
                                     test_id=test_id,
                                     x_col=cols[-1],
                                     y_col=calculated_col,
                                     y_is_calculated=True,
                                     columns_set=columns_set,
                                     show_expections=show_exceptions,
                                     display_info=display_info,
                                     f=f)

            # Also show the original features in some cases
            if test_id in ['CONSTANT_SUM', 'CONSTANT_DIFF', 'CONSTANT_PRODUCT', 'CONSTANT_RATIO']:
                self.__draw_scatter_plot(df=df,
                                         test_id=test_id,
                                         x_col=cols[0],
                                         y_col=cols[1],
                                         y_is_calculated=False,
                                         columns_set=columns_set,
                                         show_expections=show_exceptions,
                                         display_info=display_info,
                                         f=f)

        if test_id in ['LARGER_THAN_SUM', 'SIMILAR_TO_DIFF', 'LARGER_THAN_ABS_DIFF', 'SIMILAR_TO_PRODUCT',
                       'SIMILAR_TO_RATIO']:
            df = self.orig_df.copy()
            col_name_1, col_name_2, col_name_3 = cols
            if test_id in ['LARGER_THAN_SUM']:
                calculated_col = 'SUM'
                df[calculated_col] = self.numeric_vals_filled[col_name_1] + self.numeric_vals_filled[col_name_2]
            elif test_id in ['SIMILAR_TO_DIFF', 'LARGER_THAN_ABS_DIFF']:
                calculated_col = 'Absolute Difference'
                df[calculated_col] = abs(self.numeric_vals_filled[col_name_1] - self.numeric_vals_filled[col_name_2])
            elif test_id in ['SIMILAR_TO_PRODUCT']:
                calculated_col = 'PRODUCT'
                df[calculated_col] = self.numeric_vals_filled[col_name_1] * self.numeric_vals_filled[col_name_2] #df[col_name_1] * df[col_name_2]
            elif test_id in ['SIMILAR_TO_RATIO']:
                calculated_col = 'Division Results'
                df[calculated_col] = self.numeric_vals_filled[col_name_1] / self.numeric_vals_filled[col_name_2]
            self.__draw_scatter_plot(df=df,
                                     test_id=test_id,
                                     x_col=col_name_3,
                                     y_col=calculated_col,
                                     y_is_calculated=True,
                                     columns_set=columns_set,
                                     show_expections=show_exceptions,
                                     display_info=display_info,
                                     f=f)

        if test_id in ['RARE_VALUES']:
            self.__plot_count_plot(cols[0], f)

        if test_id in ['RARE_PAIRS', 'BINARY_SAME', 'BINARY_OPPOSITE', 'BINARY_IMPLIES']:
            self.__plot_heatmap(test_id, cols, f)

        if test_id in ['COLUMN_ORDERED_ASC', 'COLUMN_ORDERED_DESC', 'COLUMN_TENDS_ASC', 'COLUMN_TENDS_DESC',
                       'SIMILAR_PREVIOUS']:
            self.__draw_row_rank_plot(test_id, cols[0], show_exceptions, f)

        if test_id in ['SMALL_GIVEN_VALUE', 'LARGE_GIVEN_VALUE', 'BINARY_MATCHES_VALUE']:
            self.__draw_box_plots(test_id, cols, columns_set, f)

        if test_id in ['BINARY_MATCHES_SUM']:
            df2 = self.orig_df[cols].copy()
            df2['SUM'] = self.numeric_vals_filled[cols[0]] + self.numeric_vals_filled[cols[1]]

            fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 4))
            s = sns.boxplot(data=df2, orient='h', y=cols[2], x='SUM', ax=ax[0])
            s.set_title(f"{cols[2]} vs \nthe SUM of {cols[0]} \nand \n{cols[1]}")

            s = sns.boxplot(data=self.orig_df, orient='h', y=cols[2], x=cols[0], ax=ax[1])
            s.set_title(f"{cols[2]} vs \n{cols[0]} \nAlone")

            s = sns.boxplot(data=self.orig_df, orient='h', y=cols[2], x=cols[1], ax=ax[2])
            s.set_title(f"{cols[2]} vs \n{cols[1]} \nAlone")
            self.show_image(f)

        if test_id in ['BINARY_TWO_OTHERS_MATCH']:
            if (cols[0] in self.numeric_cols) and (cols[1] in self.numeric_cols):
                s = sns.scatterplot(data=self.orig_df, x=cols[0], y=cols[1], hue=cols[2])
                s.set_title(f'"{cols[0]}" vs "{cols[1]}"\nColor indicates "{cols[2]}"')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                self.show_image(f)

        if test_id in ['UNUSUAL_DAY_OF_WEEK']:
            sns.countplot(x=self.orig_df[cols[0]].dt.strftime('%A').fillna('NONE'))
            self.show_image(f)

        if test_id in ['UNUSUAL_DAY_OF_MONTH']:
            sns.countplot(x=self.orig_df[cols[0]].dt.day.fillna('NONE'))
            self.show_image(f)

        if test_id in ['UNUSUAL_MONTH']:
            sns.countplot(x=self.orig_df[cols[0]].dt.month.fillna('NONE'))
            self.show_image(f)

        if test_id in ['UNUSUAL_HOUR']:
            sns.countplot(x=self.orig_df[cols[0]].dt.hour.fillna('NONE'))
            self.show_image(f)

        if test_id in ['UNUSUAL_MINUTES']:
            sns.countplot(x=self.orig_df[cols[0]].dt.minute.fillna('NONE'))
            self.show_image(f)

        elif test_id in ['CONSTANT_GAP', 'LARGE_GAP', 'SMALL_GAP', 'LATER']:
            fig, ax = plt.subplots()
            gaps_arr = (self.orig_df[cols[1]] - self.orig_df[cols[0]]).dt.days.dropna()
            if gaps_arr.nunique() > 20:
                sns.histplot(x=gaps_arr)
            else:
                sns.countplot(x=gaps_arr)
            clean_x_tick_labels(fig, 1, ax)
            self.show_image(f)

        elif test_id in ['RARE_PAIRS_FIRST_CHAR']:
            df2 = self.orig_df[cols].copy()
            df2[f'{cols[0]} First Char'] = df2[cols[0]].astype(str).str[:1]
            df2[f'{cols[1]} First Char'] = df2[cols[1]].astype(str).str[:1]
            counts_data = pd.crosstab(df2[f'{cols[0]} First Char'], df2[f'{cols[1]} First Char'])
            s = sns.heatmap(counts_data, cmap="Blues", annot=True, fmt='g')
            s.set_title(f"Counts by First Characters of {cols[0]} and {cols[1]}")
            self.show_image(f)

        elif test_id in ['RARE_PAIRS_FIRST_WORD']:
            df2 = self.orig_df[cols].copy()
            col_vals = df2[cols[0]].astype(str).apply(replace_special_with_space)
            df2[f'{cols[0]} First Word'] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
            col_vals = df2[cols[1]].astype(str).apply(replace_special_with_space)
            df2[f'{cols[1]} First Word'] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
            counts_data = pd.crosstab(df2[f'{cols[0]} First Word'], df2[f'{cols[1]} First Word'])
            s = sns.heatmap(counts_data, cmap="Blues", annot=True, fmt='g')
            s.set_title(f"Counts by First Words of {cols[0]} and {cols[1]}")
            self.show_image(f)

        elif test_id in ['CORRELATED_ALPHA_ORDER']:
            df2 = self.orig_df[cols].copy()
            df2[cols[0]] = self.orig_df[cols[0]].rank(pct=True)
            df2[cols[1]] = self.orig_df[cols[1]].rank(pct=True)
            s = sns.scatterplot(data=df2, x=cols[0], y=cols[1])
            s.set_title("Values by Alphabetic Order")

            # Find the flagged values and identify them on the plot
            if show_exceptions:
                results_col_name = self.get_results_col_name(test_id, columns_set)
                results_col = self.test_results_df[results_col_name]
                flagged_idxs = np.where(results_col)
                sns.scatterplot(data=df2.loc[flagged_idxs], x=cols[0], y=cols[1], color='red', label='Flagged')
            self.show_image(f)

        elif test_id in ['LARGE_GIVEN_DATE', 'SMALL_GIVEN_DATE']:
            df2 = self.orig_df[cols].copy()
            df2[cols[1]] = df2[cols[1]].astype(float)
            if cols[1] in self.date_cols:
                df2['Epoch'] = (df2[cols[1]] - datetime.datetime(1970, 1, 1)).dt.total_seconds()
                sns.boxplot(data=df2, orient='h', y=cols[0], x='Epoch')
                plt.xlabel(cols[1])
                plt.ylabel(cols[0] + " Bin Number")
                plt.xticks([])
                plt.tight_layout()
                self.show_image(f)
            else:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
                sns.scatterplot(data=df2, x=cols[0], y=cols[1], ax=ax[0])
                for v in display_info['bin_edges']:
                    ax[0].axvline(v, color='green', linewidth=1, alpha=0.3)
                for label in ax[0].get_xmajorticklabels():
                    label.set_rotation(30)
                    label.set_horizontalalignment("right")
                ax[0].set_title('Values with Bin Edges')

                df2[cols[0]] = display_info['bin_assignments']
                s = sns.boxplot(data=df2, orient='v', x=cols[0], y=cols[1], ax=ax[1])
                s.set_xlabel(cols[0] + " Bin Number")
                s.set_title("Values by Bin")
                plt.tight_layout()
                self.show_image(f)

            # Also draw a histogram of the relevant classes.
            results_col_name = self.get_results_col_name(test_id, columns_set)
            results_col = self.test_results_df[results_col_name]
            flagged_idxs = np.where(results_col)
            flagged_df = self.orig_df.loc[flagged_idxs]
            bins_with_flagged = set(display_info['bin_assignments'].values[flagged_idxs].tolist())
            nvals = len(bins_with_flagged)
            fig, ax = plt.subplots(nrows=1, ncols=nvals, figsize=(nvals*4, 4))
            for idx, bin_id in enumerate(bins_with_flagged):
                rows_for_bin = np.where(display_info['bin_assignments'] == bin_id)
                bin_df = df2.loc[rows_for_bin]
                curr_ax = ax if nvals == 1 else ax[idx]
                s = sns.histplot(data=bin_df, x=cols[1], color='blue', bins=100, ax=curr_ax)
                s.set_title(f"Values for bin {bin_id}\nFlagged values in red")

                # Draw the flagged values
                for i in flagged_df.index:
                    if display_info['bin_assignments'][i] == bin_id:
                        curr_ax.axvline(flagged_df.loc[i, cols[1]], color='red')
            plt.tight_layout()
            self.show_image(f)

        elif test_id in ['LARGE_GIVEN_PREFIX', 'SMALL_GIVEN_PREFIX']:
            df2 = self.orig_df[cols].copy()
            col_vals = df2[cols[0]].astype(str).apply(replace_special_with_space)
            df2[cols[0]] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
            if cols[1] in self.date_cols:
                df2['Epoch'] = (df2[cols[1]] - datetime.datetime(1970, 1, 1)).dt.total_seconds()
                sns.boxplot(data=df2, orient='h', y=cols[0], x='Epoch')
                plt.xlabel(cols[1])
                plt.xticks([])
            else:
                df2[cols[1]] = df2[cols[1]].astype(float)
                sns.boxplot(data=df2, orient='h', y=cols[0], x=cols[1])
            self.show_image(f)

            # Also draw a histogram of the relevant classes.
            results_col_name = self.get_results_col_name(test_id, columns_set)
            results_col = self.test_results_df[results_col_name]
            flagged_idxs = np.where(results_col)
            flagged_df = self.orig_df.loc[flagged_idxs]
            col_vals = flagged_df[cols[0]].astype(str).apply(replace_special_with_space)
            flagged_df[cols[0]] = [x[0] if len(x) > 0 else "" for x in col_vals.str.split()]
            vals = pd.Series(flagged_df[cols[0]].astype(str).apply(replace_special_with_space))
            vals = pd.Series([x[0] if len(x) > 0 else "" for x in vals.str.split()]).unique()
            nvals = len(vals)
            fig, ax = plt.subplots(nrows=1, ncols=nvals, figsize=(nvals*4, 4))
            for v_idx, v in enumerate(vals):
                sub_df = df2[df2[cols[0]] == v]
                sub_flagged_df = flagged_df[flagged_df[cols[0]] == v]
                curr_ax = ax if nvals == 1 else ax[v_idx]
                s = sns.histplot(data=sub_df, x=cols[1], color='blue', bins=100, ax=curr_ax)
                flagged_vals = sub_flagged_df[cols[1]].values
                for fv in flagged_vals:
                    if fv is None:
                        continue
                    s.axvline(fv, color='red')
                s.set_title(f"Distribution of {cols[1]} where the first word of {cols[0]} is {v} (Flagged values in red)")
                if cols[1] in self.date_cols:
                    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
                self.show_image(f)

        elif test_id in ['SMALL_AVG_RANK_PER_ROW', 'LARGE_AVG_RANK_PER_ROW']:
            t_df = pd.DataFrame({"Avg. Percentiles": display_info['percentiles']})
            fig, ax = plt.subplots(figsize=(6, 2))
            s = sns.histplot(data=t_df, x='Avg. Percentiles')
            for fv in display_info['flagged_vals']:
                ax.axvline(fv, color='r')
            s.set_title("Mean Percentile of Numeric Values by Row")
            self.show_image(f)

        elif test_id in ['CORRELATED_GIVEN_VALUE']:
            if show_exceptions:
                results_col_name = self.get_results_col_name(test_id, columns_set)
                results_col = self.test_results_df[results_col_name]  # Array of True/False indicating the flagged rows
                flagged_idxs = np.where(results_col)

            vals = self.orig_df[cols[0]].unique()
            plotted_vals = []
            for val in vals:
                if self.orig_df[cols[0]].tolist().count(val) > 100:
                    plotted_vals.append(val)

            for val in plotted_vals:
                if (val is None) or (val != val):
                    df2 = self.orig_df[self.orig_df[cols[0]].isna()]
                else:
                    df2 = self.orig_df[self.orig_df[cols[0]] == val]
                fig, ax = plt.subplots(figsize=(3, 3))
                s = sns.scatterplot(data=df2, x=cols[1], y=cols[2], color='blue')
                s.set_title(f'Where Column "{cols[0]}" is "{val}"')

                if show_exceptions:
                    for flagged_idx in flagged_idxs:
                        if flagged_idx in list(df2.index):
                            df3 = df2.loc[flagged_idx]
                            s = sns.scatterplot(data=df3, x=cols[1], y=cols[2], color='red')
                self.show_image(f)

        elif test_id in ['GROUPED_STRINGS_BY_NUMERIC']:
            df2 = pd.DataFrame({cols[0]: self.numeric_vals_filled[cols[0]], cols[1]: self.orig_df[cols[1]]})
            if show_exceptions:
                results_col_name = self.get_results_col_name(test_id, columns_set)
                results_col = self.test_results_df[results_col_name]  # Array of True/False indicating the flagged rows
                df2['Flagged'] = results_col
                s = sns.scatterplot(data=df2, x=cols[0], y=cols[1], hue='Flagged')
            else:
                s = sns.scatterplot(data=df2, x=cols[0], y=cols[1], color='blue')
            self.show_image(f)

        elif test_id in ['LARGE_GIVEN_PAIR', 'SMALL_GIVEN_PAIR']:

            def highlight_cells():
                for row_idx in flagged_df.index:
                    v0 = flagged_df.loc[row_idx, cols[0]].strip()
                    v1 = flagged_df.loc[row_idx, cols[1]].strip()
                    patch_x = counts_df.columns.tolist().index(v1)
                    patch_y = counts_df.index.tolist().index(v0)
                    ax.add_patch(Rectangle((patch_x, patch_y), 1, 1, fill=False, edgecolor='yellow', lw=3))

            results_col_name = self.get_results_col_name(test_id, columns_set)
            results_col = self.test_results_df[results_col_name]
            flagged_idxs = np.where(results_col)
            flagged_df = self.orig_df.loc[flagged_idxs]
            vals = flagged_df[[cols[0], cols[1]]].drop_duplicates()
            nvals = len(vals)

            # Present a heatmap of the counts of each pair
            counts_df = pd.crosstab(self.orig_df[cols[0]], self.orig_df[cols[1]])
            counts_df.columns = [x.strip() for x in counts_df.columns]
            counts_df.index = [x.strip() for x in counts_df.index]
            fig_size_x = len(counts_df.columns) * 2
            fig_size_y = len(counts_df) * 0.4 if len(counts_df) > 10 else max(3, len(counts_df) * 0.9)
            fig, ax = plt.subplots(figsize=(fig_size_x, fig_size_y))
            s = sns.heatmap(counts_df, annot=True, cmap="YlGnBu", fmt='g', linewidths=1.0, linecolor='black', clip_on=False)
            s.set_title(f'Counts by combination of values in \n"{cols[0]}" and \n"{cols[1]}"')
            plt.tight_layout()
            highlight_cells()
            self.show_image(f)

            # Present a heatmap of the average value of col[2] for each pair
            if cols[2] in self.numeric_cols:
                # Use aggfunc='quantile' for date columns
                avg_df = pd.crosstab(self.orig_df[cols[0]], self.orig_df[cols[1]], values=self.numeric_vals[cols[2]], aggfunc='mean')
                fig, ax = plt.subplots(figsize=(fig_size_x, fig_size_y))
                s = sns.heatmap(avg_df, annot=True, cmap="YlGnBu", fmt='g', linewidths=1.0, linecolor='black', clip_on=False)
                s.set_title(f'Average value of \n"{cols[2]}" \nby combination of values in \n"{cols[0]}" and \n"{cols[1]}"')
                plt.tight_layout()
                highlight_cells()
                self.show_image(f)

            # Present a histogram of the relevant classes
            fig, ax = plt.subplots(nrows=1, ncols=nvals, sharey=True, figsize=(nvals*4, 4))
            for v_idx in range(nvals):
                v0 = vals.iloc[v_idx][cols[0]]
                v1 = vals.iloc[v_idx][cols[1]]
                sub_df = self.orig_df[(self.orig_df[cols[0]] == v0) & (self.orig_df[cols[1]] == v1)]
                sub_flagged_df = flagged_df[(flagged_df[cols[0]] == v0) & (flagged_df[cols[1]] == v1)]
                curr_ax = ax if nvals == 1 else ax[v_idx]
                s = sns.histplot(data=sub_df, x=cols[2], color='blue', bins=100, ax=curr_ax)
                flagged_vals = sub_flagged_df[cols[2]].values
                for fv in flagged_vals:
                    if fv is None:
                        continue
                    s.axvline(fv, color='red')
                s.set_title(f'Distribution of \n"{cols[2]}" where \n"{cols[0]}" is "{v0}" and \n"{cols[1]}" is "{v1}"')
                ax_curr = ax if nvals == 1 else ax[v_idx]
                num_ticks = len(ax_curr.xaxis.get_ticklabels())
                for label_idx, label in enumerate(ax_curr.xaxis.get_ticklabels()):
                    if label_idx != (num_ticks - 1):
                        label.set_visible(False)
            self.show_image(f)

        elif test_id in ['BINARY_RARE_COMBINATION']:
            counts_df = self.orig_df.groupby(cols).size().reset_index()
            # The 0 column is the counts. The other columns have the values from the columns in the original data.
            counts = counts_df[0]
            labels = []
            for i in counts_df.index:
                label = ''
                for c in cols:
                    label += str(counts_df.loc[i, c]) + " / "
                labels.append(label)
            _, ax = plt.subplots()
            s = sns.barplot(orient='h', y=labels, x=counts, ax=ax)
            s.set_title("Counts of combinations of values in columns")
            for count, p in zip(counts, s.patches):
                s.annotate(f'{count:.1f}', (p.get_width()+0.25, (p.get_y() + (p.get_height() / 2))+0.1))
            self.show_image(f)

        elif test_id in ['DECISION_TREE_REGRESSOR', 'LINEAR_REGRESSION'] or (test_id in ['PREV_VALUES_DT'] and cols[-1] in self.numeric_cols):
            df2 = pd.DataFrame({
                cols[-1]: self.orig_df[cols[-1]],
                'Prediction': display_info['Pred']
            })
            if show_exceptions:
                result_col_name = self.get_results_col_name(test_id, columns_set)
                results_col = self.test_results_df[result_col_name]
                df2['Flagged'] = results_col
                s = sns.scatterplot(data=df2, y='Prediction', x=cols[-1], hue='Flagged')
            else:
                s = sns.scatterplot(data=df2, y='Prediction', x=cols[-1])
            s.set_title(f'Actual vs Predicted values for "{cols[-1]}"')
            self.show_image(f)

        elif test_id in ['FIRST_WORD_SMALL_SET']:
            if len(display_info['counts']) > 1:
                s = sns.barplot(orient='h', y=display_info['counts'].index, x=display_info['counts'].values)
                self.show_image(f)


