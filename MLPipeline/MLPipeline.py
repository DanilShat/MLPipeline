import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class MLPipeline:
    def __init__(self, df):
        self.df = df

    def preprocess(self, scaler=None, scale_cols=None, encode_cols=None, handle_missing_values=None, print_outliers=False, drop_outliers=False):
        """
        Preprocesses the dataframe by scaling, encoding, handling missing values, and dealing with outliers.
    
        Parameters:
        scaler (object, optional): An instance of preprocessing scaler. Default is None.
        scale_cols (list, optional): List of column names to be scaled. Default is None.
        encode_cols (list, optional): List of column names to be label encoded. Default is None.
        handle_missing_values (dict, optional): Dictionary with column names as keys and methods ('mean', 'median', 'mode', 'drop', or a constant) as values for handling missing values. Default is None.
        print_outliers (bool, optional): If True, print the outliers in the dataframe. Default is False.
        drop_outliers (bool, optional): If True, drop the outliers in the dataframe. Default is False.
    
        Returns:
        None
        """        
        if scale_cols is not None:
            for col in scale_cols:
                self.df[col] = scaler.fit_transform(self.df[col].values.reshape(-1, 1))

        if encode_cols is not None:
            le = LabelEncoder()
            for col in encode_cols:
                self.df[col] = le.fit_transform(self.df[col])

        if handle_missing_values is not None:
            for col, method in handle_missing_values.items():
                if method == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif method == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif method == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif method == 'drop':
                    self.df = self.df[self.df[col].notna()]
                else:
                    self.df[col].fillna(method, inplace=True)

        if print_outliers:
            for col in self.df.select_dtypes(include=[np.number]).columns:
                z = np.abs(stats.zscore(self.df[col]))
                print(f"Outliers for {col}: {self.df[col][z > 3]}")

        if drop_outliers:
            for col in self.df.select_dtypes(include=[np.number]).columns:
                z = np.abs(stats.zscore(self.df[col]))
                self.df = self.df[(np.abs(stats.zscore(self.df[col])) <= 3)]
    def EDA(self, summary_statistics=False, missing_values=False, correlation_matrix=False, distribution_plots=False, pair_plots=False):
        """
        Performs Exploratory Data Analysis (EDA) on the dataframe.
    
        Parameters:
        summary_statistics (bool, optional): If True, print the summary statistics of the dataframe. Default is False.
        missing_values (bool, optional): If True, print the number of missing values in the dataframe. Default is False.
        correlation_matrix (bool, optional): If True, display the correlation matrix of the dataframe. Default is False.
        distribution_plots (bool, optional): If True, display the distribution plots of the numerical columns in the dataframe. Default is False.
        pair_plots (bool, optional): If True, display the pair plots of the numerical columns in the dataframe. Default is False.
    
        Returns:
        None
        """
        if summary_statistics:
            print("Summary Statistics:")
            print(self.df.describe())
            print("\n")

        if missing_values:
            print("Missing Values:")
            print(self.df.isnull().sum())
            print("\n")

        if correlation_matrix:
            print("Correlation Matrix:")
            corr_matrix = self.df.select_dtypes(include=[np.number]).corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
            plt.title("Correlation Matrix")
            plt.show()

        if distribution_plots:
            print("Distribution Plots:")
            for col in self.df.select_dtypes(include=[np.number]).columns:
                plt.figure(figsize=(9, 6))
                sns.histplot(data=self.df, x=col, kde=True)
                plt.title(f'Distribution of {col}')
                plt.show()
            print("\n")

        if pair_plots:
            print("Pair Plots:")
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            g = sns.pairplot(self.df[numerical_cols])
            for i, j in zip(*np.triu_indices_from(g.axes, 1)):
                g.axes[i, j].set_visible(False)
            plt.show()
            print("\n")
            
    def create_handle_missing_values(self):
        """
        Creates a dictionary for handling missing values in the dataframe.
    
        This function iterates over each column in the dataframe. If a column has any missing values, it adds the column to the dictionary with the value 'drop'.
    
        Returns:
        handle_missing_values (dict): A dictionary with column names as keys and 'drop' as values for columns with missing values.
        """
        handle_missing_values = {}

        for col in self.df.columns:
            if self.df[col].isnull().any():
                handle_missing_values[col] = 'drop'

        return handle_missing_values
    
    def choose_model(self, label_col=None):
        """
        Suggests appropriate machine learning models based on the characteristics of the dataset.
    
        Parameters:
        label_col (str, optional): The name of the label column. If None, the function will suggest models for unsupervised learning tasks. Default is None.
    
        Returns:
        None
    
        Prints:
        - The type of learning task (unsupervised, classification, regression).
        - The number of rows in the dataset.
        - Suggested models based on the size and dimensionality of the dataset.
        """
        if label_col is None:
            print("This is an unsupervised learning task.")
            print("The number of clusters is known.")
            print("Suggested models: KMeans, GaussianMixture, SpectralClustering")

            print("The number of clusters is not known.")
            print("Suggested models: DBSCAN, MeanShift, OPTICS, AffinityPropagation")

        else:
            y = self.df[label_col]
            num_rows = len(y)
            print(f"The dataset has {num_rows} rows.")

            if y.dtype.name == 'category' or len(y.unique()) < 20:  
                print("This is a classification task.")
                if num_rows < 100000:  
                    print("Suggested models for smaller datasets: SVC, RandomForestClassifier, KNeighborsClassifier, GradientBoostingClassifier")
                else:
                    print("Suggested models for larger datasets: SGDClassifier, RandomForestClassifier, LinearSVC")
            else:
                print("This is a regression task.")
                if num_rows < 100000:  
                    print("Suggested models for smaller datasets: SVR, RandomForestRegressor, KNeighborsRegressor, GradientBoostingRegressor")
                else:
                    print("Suggested models for larger datasets: SGDRegressor, RandomForestRegressor, LinearSVR")

            if self.df.select_dtypes(include=[np.number]).shape[1] > 50:  
                print("Suggested models for high dimensional datasets: LinearSVC, SGDClassifier, RandomForestClassifier" if y.dtype.name == 'category' or len(y.unique()) < 20 else "Suggested models for high dimensional datasets: LinearSVR, SGDRegressor, RandomForestRegressor")

    def DataEngineering(self, label_column, threshold=0.4, create_report=False):
        """
        Performs data engineering on the dataframe.
    
        This function creates new features by multiplying and dividing each pair of numerical columns. It also transforms numerical columns by taking the logarithm and square of the values. The function then keeps the new and transformed columns that have a correlation with the label column above a certain threshold.
    
        Parameters:
        label_column (str): The name of the label column.
        threshold (float, optional): The absolute correlation threshold for keeping new and transformed columns. Default is 0.4.
        create_report (bool, optional): If True, print the correlations of the new columns and the details of the transformed columns. Default is False.
    
        Returns:
        None
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols.remove(label_column)
        
        new_cols = pd.DataFrame(index=self.df.index)
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                new_col_mul = numerical_cols[i] + '*' + numerical_cols[j]
                new_col_div = numerical_cols[i] + '/' + numerical_cols[j]

                new_cols[new_col_mul] = self.df[numerical_cols[i]] * self.df[numerical_cols[j]]
                new_cols[new_col_div] = self.df[numerical_cols[i]] / self.df[numerical_cols[j]]

        correlations = new_cols.corrwith(self.df[label_column])

        transformed_cols = []
        original_corr = self.df[numerical_cols].corrwith(self.df[label_column])
        for col in numerical_cols:
            if self.df[col].min() > 0:
                log_col = self.df[col].apply(np.log)
                log_corr = log_col.corr(self.df[label_column])
                if abs(log_corr) > abs(original_corr[col]):
                    self.df[col + '_log'] = log_col
                    transformed_cols.append((col, col + '_log', original_corr[col], log_corr))
            square_col = self.df[col].apply(np.square)
            square_corr = square_col.corr(self.df[label_column])
            if abs(square_corr) > abs(original_corr[col]):
                self.df[col + '_squared'] = square_col
                transformed_cols.append((col, col + '_squared', original_corr[col], square_corr))

        cols_to_keep = correlations[correlations.abs() > threshold].index.tolist()

        self.df = pd.concat([self.df, new_cols[cols_to_keep]], axis=1)

        if create_report:
            print("New columns and their correlations:")
            print(correlations[correlations.abs() > threshold])
            print("\nTransformed columns added:")
            for old_col, new_col, old_corr, new_corr in transformed_cols:
                print(f"{old_col} ---> {new_col}: {abs(old_corr)} ---> {abs(new_corr)}")
    def split_data(self, input_cols, label_col, split=0.2, shuffle=True, threshold=None):
        """
        Splits the dataframe into training and testing sets.

        Parameters:
        input_cols (list): List of column names to be used as input features.
        label_col (str): The name of the label column.
        split (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
        shuffle (bool, optional): Whether or not to shuffle the data before splitting. If False, the method does a sequential split. Default is True.
        threshold (float, optional): The absolute value of the correlation coefficient threshold for feature selection. Default is None.

        Returns:
        X_train (DataFrame): The input features for the training set.
        X_test (DataFrame): The input features for the testing set.
        y_train (Series): The labels for the training set.
        y_test (Series): The labels for the testing set.
        """
        if threshold is not None:
            correlations = self.df[input_cols].corrwith(self.df[label_col]).abs()
            input_cols = correlations[correlations > threshold].index.tolist()
            print(f"Columns in X_train: {input_cols}")
            print(f"Columns not in X_train: {list(set(self.df.columns) - set(input_cols))}")

        X = self.df[input_cols]
        y = self.df[label_col]

        if shuffle:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=True)
        else:
            train_size = int(len(self.df) * (1 - split))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

        return X_train, X_test, y_train, y_test
    
    def execute(self, config):
        """
        Executes methods of the class based on the provided configuration.
    
        Parameters:
        config (dict): A dictionary where keys are method names and values are dictionaries of parameters for the corresponding method.
    
        Returns:
        None
        """
        for method, params in config.items():
            getattr(self, method)(**params)
    
    def default_config(self):
        """
        Returns the default configuration for the class methods.
    
        Returns:
        config (dict): A dictionary where keys are method names and values are dictionaries of default parameters for the corresponding method.
        """
        return {
            'preprocess': {'scaler' : None, 'scale_cols' : None, 'encode_cols' : None, 'handle_missing_values' : None, 'print_outliers' : False, 'drop_outliers': False},
            'EDA': {'summary_statistics' : False, 'missing_values' : False, 'correlation_matrix' : False, 'distribution_plots' : False, 'pair_plots' : False},
            'choose_model' : {'label_col' : None},
            'DataEngineering': {'label_column' : 'label', 'threshold' : 0.4, 'create_report' : False}
        }
