## Import Packages ##

import pandas as pd

import numpy as np

import scipy.stats

from scipy import stats

from scipy.stats import ttest_ind, mannwhitneyu

from scipy.stats import norm

 

 

## Tools for statistical tests ##

class StatsTool:

   

    # Initialize class

    def __init__(self):

        pass

   

    

    #Get results of statistical test

    def stat_results(self,

                     p_value: float,

                     alternative: str,

                     alpha: float = 0.05,

                     metric: str = None,

                     print_results: bool = False) -> bool:       

        '''

        Description:

            - Provide results from statistical test.

 

        Parameters:

            - p_value (float): The numerical value representing the p_value.

            - alternative (str): What type of test (two-sided, less, greater)?

            - alpha (float): The numerical value representing the level of significance. It represents the acceptable probability of making a Type 1 error (reject hypotheses when actually true)

            - metric (str): The name of the metric being tested. 

            - print_results (bool): Indicates if the results will be printed.

           

        Output:

            significant (bool): If the statistical test is significant.

        '''

 

        #Confidence

        conf = int((1 - alpha) * 100)

 

        #Two-sided test

        if (alternative == 'two-sided'):

            #Ho: D1 == D2

            #Ha: D1 != D2

            if p_value > alpha:

                #Ho

                if print_results == True:

                    print(f'\nNO statistical significance: There is insufficient evidence that the control group {metric} are different than the treatment group {metric}')

                significant = False

            else:

                #Ha

                if print_results == True:

                    print(f'\nStatistical significance: We are at least {conf}% confidence that the control group {metric} are different than the treatment group {metric}')  

                significant = True

 

        #One-sided test: Ha: D1 < D2

        elif (alternative == 'less'):

            #Ho: D1 = D2

            #Ha: D1 < D2

            if p_value > alpha:

                #Ho

                if print_results == True:

                    print(f'\nNO statistical significance: There is insufficient evidence that the treatment group {metric} is less than the control group {metric}.')

                significant = False

            else:

                #Ha

                if print_results == True:

                    print(f'\nStatistical significance: We are at least {conf}% confidence that the treatment group {metric} is less than the control group {metric}.')   

                significant = True

 

        #One-sided test: Ha: D1 > D2

        elif (alternative == 'greater'):

            #Ho: D1 = D2

            #Ha: D1 > D2

            if p_value > alpha:

                #Ho

                if print_results == True:

                    print(f'NO statistical significance: There is insufficient evidence that the treatment group {metric} is greater than the control group {metric}.')

                significant = False

            else:

                #Ha

                if print_results == True:

                    print(f'\nStatistical significance: We are at least {conf}% confidence that that the treatment group {metric} is greater than the control group {metric}.')

                significant = True

                               

        return(significant)

                   

 

    def two_sample_t_test_for_means(self,

                                    df1: pd.DataFrame,

                                    df2: pd.DataFrame,

                                    col: str,

                                    alternative: str,

                                    alpha: float = 0.05,

                                    metric: str = None,

                                    print_results: bool = False,

                                    **kwargs) -> dict:

        '''

        Description: Perform an independent two-sample t-test using scipy.stats.ttest_ind.

 

        Data Assumptions:

            - The two samples are **independent** of each other.

            - Data in each group is **approximately normally distributed**, especially important for small sample sizes.

            - Data should be **continuous and numeric**.

            - For the standard t-test (`equal_var=True`), the variances of the two groups should be **equal**.

              If this assumption is violated, use `equal_var=False` to perform Welch's t-test, which does not assume equal variances.

 

        Parameters:

            - df1 (pd.DataFrame): First sample data (treatment group).

            - df2 (pd.DataFrame): Second sample data (treatment group).

            - col (str): Column of interest

            - alternative (str): What type of test (two-sided, less, greater)?

            - alpha (float): The numerical value representing the level of significance. It represents the acceptable probability of making a Type 1 error (reject hypotheses when actually true)

            - metric (str): The metric being tested.

            - print_results (bool): Indicates if the results will be printed.

            **kwargs (Any): Additional keyword arguments passed to scipy.stats.ttest_ind.

           

        Output:

            - Dictionary of results including statistical significance, p-value, test statistics, and sample sizes.

        '''

 

        #Run t-test

        t_test_results = ttest_ind(df1[col], df2[col], alternative = alternative, **kwargs)

 

        #Get p_value

        p_value = t_test_results.pvalue

 

        #Results

        significance = self.stat_results(p_value = p_value,

                                         alternative = alternative,

                                         alpha = alpha,

                                         metric = metric,

                                         print_results = print_results)

               

        #Results in dictionary form

        results_dict = {'significant': significance,

                        'p_value': p_value,

                        'test_statistic': t_test_results.statistic,

                        'df': t_test_results.df,

                        'sample_size_1': len(df1[col].notna()),

                        'sample_size_2': len(df2[col].notna())}

       

        return(results_dict)

 

 

    def two_sample_non_parametric_test(self,

                                       df1: pd.DataFrame,

                                       df2: pd.DataFrame,

                                       col: str,

                                       alternative: str,

                                       alpha: float = 0.05,

                                       metric: str = None,

                                       print_results: bool = False,

                                       **kwargs) -> dict:

        '''   

        Description:

            Perform a Mann-Whitney U test (Wilcoxon rank-sum test) using scipy.stats.mannwhitneyu.

 

        Data Assumptions:

            - The two samples are **independent** of each other.

            - Data must be **ordinal or continuous** (but not necessarily normally distributed).

            - Observations are **rankable** (i.e., you can meaningfully say one value is greater or less than another).

            - The test compares **distributions** — not necessarily means — and is sensitive to **differences in medians**.

            - If both samples have **identical values**, the test may raise a warning or return a p-value of 1.0.

 

        Parameters:

            - df1 (pd.DataFrame): First sample data (treatment group).

            - df2 (pd.DataFrame): Second sample data (control group).

            - col (str): Column of interest

            - alternative (str): What type of test (two-sided, less, greater)?

            - alpha (float): The numerical value representing the level of significance. It represents the acceptable probability of making a Type 1 error (reject hypotheses when actually true)

            - metric (str): The metric being tested.

            - print_results (bool): Indicates if the results will be printed.

            - **kwargs (Any): Additional keyword arguments passed to scipy.stats.ttest_ind.

           

        Output:

            - Dictionary of results including statistical significance, p-value, test statistics, and sample sizes.

 

        Notes:

            - This is a non-parametric test and does not assume normality or equal variances.

            - It is less powerful than the t-test when the normality assumption holds, but more robust when it doesn't.

        '''

 

        # Remove Null values and make sure they are numeric

        group1 = df1.loc[df1[col].notna()][col].astype(float)

        group2 =  df2.loc[df2[col].notna()][col].astype(float)

 

        # Mann–Whitney U test

        #No assumption about normality of distribution

        #Tests differences in the distributions through a rank-sumed methodology (accounts for outliers)

        stat, p_value = mannwhitneyu(group1, group2, alternative = alternative, **kwargs)

 

        #Results

        significance = self.stat_results(p_value = p_value,

                                         alternative = alternative,

                                         alpha = alpha,

                                         metric = metric,

                                         print_results = print_results)

       

        #Results in dictionary form

        results_dict = {'significant': significance,

                        'p_value': p_value,

                        'test_statistic': stat,

                        'sample_size_1': len(group1),

                        'sample_size_2': len(group2)}

       

        return(results_dict)

 

 

    def two_sample_z_test_proportions(self,

                                      df1: pd.DataFrame,

                                      df2: pd.DataFrame,

                                      col: str,

                                      alternative: str,

                                      alpha: float = 0.05,

                                      print_results: bool = False,

                                      metric: str = None) -> dict:

        '''

        Description:

            Perform a two-sample z-test for comparing proportions.

 

        Data Assumptions:

            - The samples are **independent**.

            - The outcome is **binary** (e.g., success/failure).

            - Each group has a **fixed number of trials** (nobs).

            - Data comes from a **random sample** of the population.

            - **Normal approximation** is valid:

                Each group should satisfy:

                    n * p ≥ 5 and n * (1 - p) ≥ 5

                where p is the observed proportion of successes.

                If not, results may be inaccurate and an exact test (e.g., Fisher's) may be preferred.

 

        Parameters:

            - df1 (pd.DataFrame): First sample data (treatment group).

            - df2 (pd.DataFrame): Second sample data (control group).

            - col (str): Column of interest

            - alternative (str): What type of test (two-sided, less, greater)?

            - alpha (float): The numerical value representing the level of significance. It represents the acceptable probability of making a Type 1 error (reject hypotheses when actually true)

            - metric (str): The metric being tested.

            - print_results (bool): Indicates if the results will be printed.

           

        Output:

            - Dictionary of results including statistical significance, p-value, test statistics, and sample sizes.

        '''

        #Proportions

        p1_hat = df1[col].mean()

        p2_hat = df2[col].mean()

 

        #Sample sizes

        n1 = len(df1.loc[df1[col].notna()][col])

        n2 = len(df2.loc[df2[col].notna()][col])

        n = n1 + n2

       

        #P-star

        #Average over both samples (estimate of population proportion)

        p_star = (df1[col].sum() + df2[col].sum()) / (n1 + n2)

       

        #Evaluate assumption

        if ((n * p_star) < 5) & ((n * (1 - p_star)) < 5):

            logging.error('Error: This does not meet the required conditions for this test. It did not meet this condition: n * p ≥ 5 and n * (1 - p) ≥ 5')

 

        #Calculate z-score

        z_score = (p1_hat - p2_hat) / np.sqrt(p_star * (1 - p_star) * ((1 / n1) + (1 / n2)))

 

        # Two-sided z-test

        if alternative == 'two-sided':

            #Ho: P1 = P2

            #Ha: P1 != P2

            p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z_score))) #two-sided

 

        # One-sided z-test

        if alternative == 'less':

            #Ho: P1 = P2

            #Ha: P1 < P2

            p_value = scipy.stats.norm.cdf(z_score)

 

        #One sided z-test

        if alternative == 'greater':

            #Ho: P1 = P2

            #Ha: P1 > P2

            p_value = 1 - scipy.stats.norm.cdf(z_score)

 

        #Results

        significance = self.stat_results(p_value = p_value,

                                         alternative = alternative,

                                         alpha = alpha,

                                         metric = metric,

                                         print_results = print_results)

               

        #Results in dictionary form

        results_dict = {'significant': significance,

                        'p_value': p_value,

                        'test_statistic': z_score,

                        'sample_size_1': n1,

                        'sample_size_2': n2

                       }

       

        return(results_dict)

   

    

    #This code was developed by Evan Spence (no longer at Allstate)

    def calculate_sample_size_duration_continuous(self,

                                                  blv,

                                                  std_dev,

                                                  mde,

                                                  samples_per_day,

                                                  power = 0.8,

                                                  mde_type='absolute'):

        """

        Calculate the required sample size and total duration for a continuous metric A/B test, where

        samples_per_day is the total number of samples collected for both groups.

 

        Parameters:

        - blv: Baseline value of the metric being tested (e.g., average revenue per user).

        - std_dev: Standard deviation of the metric.

        - mde: Minimum detectable effect, specified either as a percentage (relative to blv) or an absolute value.

        - samples_per_day: Expected total number of samples (participants) per day overall (including both groups).

        - alpha: Significance level (probability of Type I error).

        - power: Test power (1 - probability of Type II error).

        - mde_type: Indicates whether mde is 'percentage' or 'absolute'.

 

        Returns:

        - Total sample size required (including both groups).

        - Total duration in days to achieve the required sample size for both groups.

        """

 

        #Get alpha value

        alpha = self.alpha

       

        #Calculate the z-score of alpha and beta

        z_alpha = norm.ppf(1 - alpha / 2)

        z_beta = norm.ppf(power)

 

        # Convert MDE from percentage to absolute if necessary

        if mde_type == 'percentage':

            mde_absolute = (mde / 100.0) * blv

        else:

            mde_absolute = mde

 

        # Calculate the sample size using the formula for difference in means

        sample_size_per_group = ((z_alpha + z_beta) ** 2) * (2 * (std_dev ** 2)) / (mde_absolute ** 2)

 

        # Round up to the nearest whole number

        sample_size_per_group = np.ceil(sample_size_per_group)

 

        # Calculate the total sample size required for both groups

        total_sample_size = sample_size_per_group * 2

 

        # Calculate the total duration in days to achieve the sample size for both groups

        total_days = np.ceil(total_sample_size / samples_per_day)

 

        # Print statement for input variables

        print(f"""

        Input Variables:

        Baseline: {blv}

        Standard Deviation: {std_dev}

        MDE: {mde} ({mde_type})

        Samples per Day: {samples_per_day}

        Alpha: {alpha}

        Power: {power}

        \n""")

 

        return int(total_sample_size), int(total_days)

   

    

    #This code was developed by Evan Spence (no longer at Allstate)

    def calculate_pooled_std_dev(self,

                                 counts,

                                 means,

                                 std_devs):

        """

        Calculate the pooled standard deviation for combined groups.

 

        Parameters:

        - counts: An array of the total counts for each group.

        - means: An array of the mean values for each group.

        - std_devs: An array of the standard deviations for each group.

 

        Returns:

        - The pooled standard deviation of all groups combined.

        """

        # Ensure input arrays are numpy arrays for element-wise operations

        counts = np.array(counts)

        means = np.array(means)

        std_devs = np.array(std_devs)

 

        # Calculate the grand mean

        grand_mean = np.sum(counts * means) / np.sum(counts)

 

        # Calculate the variance within each group

        var_within = np.sum((counts - 1) * std_devs**2)

 

        # Calculate the variance between group means

        var_between = np.sum(counts * (means - grand_mean)**2) / len(counts)

 

        # Calculate the pooled variance, and then take the square root for standard deviation

        pooled_variance = (var_within + var_between) / (np.sum(counts) - len(counts))

        pooled_std_dev = np.sqrt(pooled_variance)

 
        return (pooled_std_dev)
