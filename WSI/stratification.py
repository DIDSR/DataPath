# -*- coding: utf-8 -*-
"""
Stratification and Visualization Utilities for Dataset Splitting

Created on Mon Apr  7 10:54:59 2025

@author: Tanviben.Patel
"""
"""
---------------------------------------------------------------------------
Created on Mon Apr  7 10:54:59 2025

----------------------------------------------------------------------------

**Title:**        DataPath Toolbox - WSI Handler module

**Description:**  This is the WSI Handler module for the DataPath toolbox. It is includes stratification class and several methods
              
**Classes:**      stratification          

This module provides a class `stratification`  to perform random and stratified train/val/test splitting
on image datasets with hierarchical class structure (category and subtype).
    
---------------------------------------------------------------------------
Author: Tanviben.Patel (tanviben.patel@fda.hhs.gov) SeyedM.MousaviKahaki (seyed.kahaki@fda.hhs.gov)
Version ='1.0'
---------------------------------------------------------------------------
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


class stratification:
    """
    A class to perform random and stratified train/val/test splitting
    on image datasets with hierarchical class structure (category and subtype).

    Attributes
    ----------
    root_dir : str
        Root directory of the dataset.
    categories : dict
        Dictionary defining category and subcategory structure.
    test_size : float
        Proportion of dataset to include in the test split.
    val_size : float
        Proportion of remaining dataset to include in the validation split.
    df : pd.DataFrame
        DataFrame containing image paths and labels.
    """

    def __init__(self, root_dir: str, categories: dict, test_size=0.2, val_size=0.25):
        """
        Initialize the Stratification object and load the dataset.

        Parameters
        ----------
        root_dir : str
            Root path to the dataset.
        categories : dict
            Dictionary with keys as category names and values as dicts with
            'path' and 'subcategories' list.
        test_size : float, optional
            Fraction of data to reserve for testing, by default 0.2.
        val_size : float, optional
            Fraction of training data to reserve for validation, by default 0.25.
        """
        self.root_dir = root_dir
        self.categories = categories
        self.test_size = test_size
        self.val_size = val_size
        self.df = self._load_data()

    def _load_data(self):
        """
        Load image paths and corresponding labels into a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: image_path, category, subtype
        """
        data_records = []
        for category, details in self.categories.items():
            category_path = details["path"]
            subcategories = details["subcategories"]

            if not os.path.exists(category_path):
                print(f"Skipping {category_path}, path not found!")
                continue

            for subtype in subcategories:
                subtype_path = os.path.join(category_path, subtype)
                if not os.path.exists(subtype_path):
                    continue

                image_files = [
                    os.path.join(subtype_path, img)
                    for img in os.listdir(subtype_path)
                    if img.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]

                for img_path in image_files:
                    data_records.append([img_path, category, subtype])

        df = pd.DataFrame(data_records, columns=["image_path", "category", "subtype"])
        if df.empty:
            raise ValueError("No images found. Check dataset paths.")
        return df

    def split_random_by_category(self):
        """
        Perform random train/val/test split based on category (not stratified).

        Returns
        -------
        tuple of pd.DataFrame
            X_train, X_val, X_test DataFrames.
        """
        X_train, X_test = train_test_split(self.df, test_size=self.test_size, stratify=None)
        X_train, X_val = train_test_split(X_train, test_size=self.val_size, stratify=None)
        return X_train, X_val, X_test

    def split_stratified_by_category(self):
        """
        Perform stratified train/val/test split based on category labels.

        Returns
        -------
        tuple of pd.DataFrame
            X_train, X_val, X_test DataFrames.
        """
        X_train, X_test = train_test_split(self.df, test_size=self.test_size, stratify=self.df["category"], random_state=42)
        X_train, X_val = train_test_split(X_train, test_size=self.val_size, stratify=X_train["category"], random_state=42)
        return X_train, X_val, X_test

    def split_random(self):
        """
        Perform random train/val/test split based on subtype (not stratified).

        Returns
        -------
        tuple of pd.DataFrame
            X_train, X_val, X_test DataFrames.
        """
        X_train, X_test = train_test_split(self.df, test_size=self.test_size, stratify=None, random_state=42)
        X_train, X_val = train_test_split(X_train, test_size=self.val_size, stratify=None, random_state=42)
        return X_train, X_val, X_test

    def split_stratified(self):
        """
        Perform stratified train/val/test split based on subtype labels.

        Returns
        -------
        tuple of pd.DataFrame
            X_train, X_val, X_test DataFrames.
        """
        X_train, X_test = train_test_split(self.df, test_size=self.test_size, stratify=self.df["subtype"], random_state=42)
        X_train, X_val = train_test_split(X_train, test_size=self.val_size, stratify=X_train["subtype"], random_state=42)
        return X_train, X_val, X_test

    @staticmethod
    def plot_category_distribution(df, title="Category Distribution"):
        """
        Plot bar chart of category distribution with counts and percentages.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'category' column.
        title : str, optional
            Title of the plot, by default "Category Distribution"
        """
        category_counts = df['category'].value_counts()
        total = category_counts.sum()

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=category_counts.index, y=category_counts.values, palette="Set2")
        for i, count in enumerate(category_counts.values):
            percent = (count / total) * 100
            ax.text(i, count + 1, f"{count} ({percent:.1f}%)", ha='center', fontsize=12)
        plt.title(title)
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_category_split_distribution(split_dict, title="Category Distribution by Split"):
        """
        Plot category-level distribution across train/val/test splits with counts and percentages.

        Parameters
        ----------
        split_dict : dict
            Dictionary with keys 'Train', 'Validation', 'Test' and value counts.
        title : str, optional
            Title of the plot, by default "Category Distribution by Split"
        """
        labels = list(split_dict["Train"].keys())
        x = np.arange(len(labels))

        train_total = sum(split_dict["Train"].values())
        val_total = sum(split_dict["Validation"].values())
        test_total = sum(split_dict["Test"].values())

        train_pct = [v / train_total * 100 for v in split_dict["Train"].values()]
        val_pct = [v / val_total * 100 for v in split_dict["Validation"].values()]
        test_pct = [v / test_total * 100 for v in split_dict["Test"].values()]

        plt.figure(figsize=(10, 5))
        bar1 = plt.bar(x, train_pct, width=0.3, label='Train')
        bar2 = plt.bar(x + 0.3, val_pct, width=0.3, label='Validation')
        bar3 = plt.bar(x + 0.6, test_pct, width=0.3, label='Test')

        for bars, counts in zip([bar1, bar2, bar3], [split_dict["Train"], split_dict["Validation"], split_dict["Test"]]):
            for bar, (label, count) in zip(bars, counts.items()):
                height = bar.get_height()
                text = f"{count} ({height:.1f}%)"
                plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, text, ha='center', va='bottom', fontsize=10)

        plt.xticks(x + 0.3, labels)
        plt.ylabel("Percentage")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_subtype_distribution(splits_dict, title="Subtype Distribution"):
        """
        Plot subtype-level distribution across train/val/test splits in percentages.

        Parameters
        ----------
        splits_dict : dict
            Dictionary with keys 'Train', 'Validation', 'Test' and values as subtype counts.
        title : str, optional
            Title of the plot, by default "Subtype Distribution"
        """
        labels = list(splits_dict["Train"].keys())
        x = np.arange(len(labels))

        train_total = sum(splits_dict["Train"].values())
        val_total = sum(splits_dict["Validation"].values())
        test_total = sum(splits_dict["Test"].values())

        train_pct = [v / train_total * 100 for v in splits_dict["Train"].values()]
        val_pct = [v / val_total * 100 for v in splits_dict["Validation"].values()]
        test_pct = [v / test_total * 100 for v in splits_dict["Test"].values()]

        plt.figure(figsize=(14, 6))
        bar1 = plt.bar(x, train_pct, width=0.3, label='Train')
        bar2 = plt.bar(x + 0.3, val_pct, width=0.3, label='Validation')
        bar3 = plt.bar(x + 0.6, test_pct, width=0.3, label='Test')

        for bars in [bar1, bar2, bar3]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}%", ha='center', va='bottom')

        plt.xticks(x + 0.3, labels, rotation=45)
        plt.ylabel("Percentage")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
