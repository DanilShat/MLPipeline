# MLPipeline

## Overview
MLPipeline is a Python package designed to streamline the process of building machine learning pipelines. It's primarily intended for personal use, but others may find it useful. The goal of this package is to speed up common machine learning tasks such as preprocessing, exploratory data analysis (EDA), data engineering, and preparing data for modeling.

## Features
- **Preprocessing**: Simplify the process of cleaning and preparing your data for analysis.
- **EDA**: Quickly generate exploratory analyses to gain insights from your data.
- **Data Engineering**: Easily engineer features to improve model performance.
- **Model Preparation**: Prepare your data for modeling with just a few lines of code.

## Usage
Users can use methods from this class independently, or they can do everything by just creating a configuration.
Configuration example:

config = {'preprocess': {'scaler': None,
  'scale_cols': None,
  'encode_cols': None,
  'handle_missing_values': pipeline.create_handle_missing_values(),
  'print_outliers': True,
  'drop_outliers': True},
 'EDA': {'summary_statistics': True,
  'missing_values': False,
  'correlation_matrix': True,
  'distribution_plots': False,
  'pair_plots': True},
 'choose_model': {'label_col': 'Quality'},
 'DataEngineering': {'label_column': 'Quality',
  'threshold': 0.2,
  'create_report': True}}

