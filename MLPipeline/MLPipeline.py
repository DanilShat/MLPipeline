from sklearn.preprocessing import LabelEncoder
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

class MLPipeline:
    def __init__(self, df):
        self.df = df

    def preprocess(self, scaler=None, scale_cols=None, encode_cols=None, handle_missing_values=None, print_outliers=False, drop_outliers=False):
        # If there are columns to scale
        if scale_cols is not None:
            for col in scale_cols:
                self.df[col] = scaler.fit_transform(self.df[col].values.reshape(-1, 1))

        # If there are columns to label encode
        if encode_cols is not None:
            le = LabelEncoder()
            for col in encode_cols:
                self.df[col] = le.fit_transform(self.df[col])

        # If there are missing values to handle
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

        # If print_outliers is True
        if print_outliers:
            for col in self.df.select_dtypes(include=[np.number]).columns:
                z = np.abs(stats.zscore(self.df[col]))
                print(f"Outliers for {col}: {self.df[col][z > 3]}")

        # If drop_outliers is True
        if drop_outliers:
            for col in self.df.select_dtypes(include=[np.number]).columns:
                z = np.abs(stats.zscore(self.df[col]))
                self.df = self.df[(np.abs(stats.zscore(self.df[col])) <= 3)]
    def summary_statistics(self):
        return self.df.describe()
    def missing_values(self):
        return self.df.isnull().sum()
    def correlation_matrix(self):
        corr_matrix = self.df.corr()
        return corr_matrix
    def EDA(self, summary_statistics=False, missing_values=False, correlation_matrix=False, distribution_plots=False, pair_plots=False):
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
            # Select only the numerical columns
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            # Create pair plots
            sns.pairplot(self.df[numerical_cols])
            plt.show()
            print("\n")
    def create_handle_missing_values(self):
        # Initialize an empty dictionary
        handle_missing_values = {}

        # Iterate over each column in the DataFrame
        for col in self.df.columns:
            # If the column has any missing values
            if self.df[col].isnull().any():
                # Add the column to the dictionary with the value 'drop'
                handle_missing_values[col] = 'drop'

        return handle_missing_values
    
    def choose_model(self, label_col=None):
        # Step 1: Determine if it's supervised or unsupervised learning
        if label_col is None:
            print("This is an unsupervised learning task.")
            print("The number of clusters is known.")
            print("Suggested models: KMeans, GaussianMixture, SpectralClustering")

            print("The number of clusters is not known.")
            print("Suggested models: DBSCAN, MeanShift, OPTICS, AffinityPropagation")

        # Step 2: If it's supervised learning, check if it's a regression or classification task
        else:
            y = self.df[label_col]
            if y.dtype.name == 'category' or len(y.unique()) < 20:  # Change this threshold based on your needs
                print("This is a classification task.")
            else:
                print("This is a regression task.")

            num_rows = len(y)
            print(f"The dataset has {num_rows} rows.")

            if num_rows < 100000:  # Change this threshold based on your needs
                print("Suggested models for smaller datasets: SVC, RandomForestClassifier, KNeighborsClassifier, GradientBoostingClassifier")
            else:
                print("Suggested models for larger datasets: SGDClassifier, RandomForestClassifier, LinearSVC")

            if self.df.select_dtypes(include=[np.number]).shape[1] > 50:  # If there are more than 50 numerical features
                print("Suggested models for high dimensional datasets: LinearSVC, SGDClassifier, RandomForestClassifier")
    def DataEngineering(self, label_column, threshold=0.4, create_report=False):
        # Get numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target column
        numerical_cols.remove(label_column)
        
        # Apply data engineering methods to every numerical column
        new_cols = pd.DataFrame(index=self.df.index)
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                new_col_mul = numerical_cols[i] + '*' + numerical_cols[j]
                new_col_div = numerical_cols[i] + '/' + numerical_cols[j]

                new_cols[new_col_mul] = self.df[numerical_cols[i]] * self.df[numerical_cols[j]]
                new_cols[new_col_div] = self.df[numerical_cols[i]] / self.df[numerical_cols[j]]

        # Calculate correlations with target column for new columns
        correlations = new_cols.corrwith(self.df[label_column])

        # Apply transformations to original columns and calculate their correlations
        transformed_cols = []
        original_corr = self.df[numerical_cols].corrwith(self.df[label_column])
        for col in numerical_cols:
            # Apply log transform where possible
            if self.df[col].min() > 0:
                log_col = self.df[col].apply(np.log)
                log_corr = log_col.corr(self.df[label_column])
                # Add transformed column if its correlation with target is higher than original column
                if abs(log_corr) > abs(original_corr[col]):
                    self.df[col + '_log'] = log_col
                    transformed_cols.append((col, col + '_log', original_corr[col], log_corr))
            # Apply square transform
            square_col = self.df[col].apply(np.square)
            square_corr = square_col.corr(self.df[label_column])
            # Add transformed column if its correlation with target is higher than original column
            if abs(square_corr) > abs(original_corr[col]):
                self.df[col + '_squared'] = square_col
                transformed_cols.append((col, col + '_squared', original_corr[col], square_corr))

        # Filter new columns with abs of correlation more than threshold
        cols_to_keep = correlations[correlations.abs() > threshold].index.tolist()

        # Update dataframe to keep only relevant new columns and original columns
        self.df = pd.concat([self.df, new_cols[cols_to_keep]], axis=1)

        # Create report if specified
        if create_report:
            print("New columns and their correlations:")
            print(correlations[correlations.abs() > threshold])
            print("\nTransformed columns added:")
            for old_col, new_col, old_corr, new_corr in transformed_cols:
                print(f"{old_col} ---> {new_col}: {abs(old_corr)} ---> {abs(new_corr)}")


