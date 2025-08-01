##################################################################

### Import packages

##################################################################

import sys

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, force=True)


   

##################################################################

### Install Python Package

##################################################################

def install(package: str) -> None:

    """

    This function allows the user to install packages, even in a Python script.

   

    Parameters:

        - package = the name of the Python package you wish to install, in quotes.

    """

   

    #Git package is not named unless in a list

    package_is_name = False


    #Named packages, common Git packages installed
    named_packages = {'package_name': 'https://$GITHUB_USER:$GITHUB_PAT@github.com/package_name.git@master'}
   

    #Get Git url based on named packages
    if package.lower() in list(named_packages.keys()):

        package = named_packages[package.lower()]

        package_is_name = True
   

    #Installing Git package

    if 'github.com' in package:

        package_name = package.split('/')[-1].split('.git')[0]


        #Retrieve Git credentials (change from string to actual credentials)

        try:

            user = package.split('$')[1].split('@')[0].split(':')[0]

            pwd = package.split('$')[2].split('@')[0].split(':')[0]

       

        #Error handling

        except:

            if package_is_name == False:

                raise Exception("ERROR: GIT CREDENTIALS NOT FOUND")

            if package_is_name == True:

                raise Exception("ERROR: GIT CREDENTIALS NOT FOUND: ADD GITHUB_PAT AND GITHUB_USER TO YOUR USER ENVIRONMENT")

       

        logger.info(f"INSTALLING GIT PACKAGE: {package_name}")

       

        #Install Git package

        subprocess.run([sys.executable, '-m', 'pip', 'install', 'git+' + package.replace(user, os.getenv(user)).replace(pwd, os.getenv(pwd)).replace('$', ''), '--user'])

       

        try:

            #Append package site (otherwise kernel restart is required)

            sys.path.append('/home/ubuntu/.local/lib/python3.9/site-packages')

        except:

            pass

       

        logger.info(f"SUCCESSFULLY INSTALLED GIT PACKAGE: {package_name}")

   

    #Install Python package

    else:

        logger.info(f"INSTALLING PACKAGE: {package}")

       

        #Install package

        subprocess.run([sys.executable, "-m", "pip", "install", package, '--user'])

       

        logger.info(f"SUCCESSFULLY INSTALLED PACKAGE: {package}")

       

    #Restart program- used to make sure package is available after it is download

 

 

   

    

##################################################################

### Import The Rest of the Required Packages

##################################################################

import numpy as np

import sys

import os

import subprocess

from datetime import datetime

from itertools import product

 

##Packages that may not be installed yet##

try:

    import pandas as pd

except:

    install('pandas')

    import pandas as pd

 

 

 

##################################################################

### Error Testing: Test that column is in Pandas dataframe

##################################################################

def error_col_in_df(df: pd.DataFrame,

                    col: str) -> None:

    """

    This function tests that the inputted column is in the Pandas dataframe.

   

    Parameters:

        - df (pd.DataFrame): DataFrame containing the state column.

        - col (str): Inputted column

    """       

    #Check that the column is in the dataframe

    if col not in df.columns:

        raise ValueError(f"ERROR: Column '{col}' not found.")

        

    return()

   

    

    

##################################################################

### Error Testing: Test that column is a string

##################################################################

def error_test_string_col(df: pd.DataFrame,

                          col: str) -> None:

    """

    This function tests that the inputted column is a string

   

    Parameters:

        - df (pd.DataFrame): DataFrame containing the state column.

        - col (str): Inputted column

    """   

    #Check that column is in dataframe

    error_col_in_df(df, col)

       

    #Check that the column is a string

    if not pd.api.types.is_string_dtype(df[col]):

        raise ValueError(f"ERROR: Column '{col}' is not of string type.")

       

    return()

 

 

 

##################################################################

### Error Testing: Test that column is numeric

##################################################################

def error_test_num_col(df: pd.DataFrame,

                       col: str) -> None:

    """

    This function tests that the inputted column is a string

   

    Parameters:

        - df (pd.DataFrame): DataFrame containing the state column.

        - col (str): Inputted column

    """   

    #Check that column is in dataframe

    error_col_in_df(df, col)

       

    #Check that the column is a string

    if not pd.api.types.is_numeric_dtype(df[col]):

        raise ValueError(f"ERROR: Column '{col}' is not of string type.")

       

    return()

 

 

 

##################################################################

### Dictinary of States and Their Abbreviations

##################################################################

def state_abbrev_dict() -> dict[str, str]:

    """

    This function will provide a dictionary of US states with their corresponding abbreviations.

    """

       

    #List of states

    state2abbrev = {

        'ALASKA': 'AK',

        'ALABAMA': 'AL',

        'ARKANSAS': 'AR',

        'ARIZONA': 'AZ',

        'CALIFORNIA': 'CA',

        'COLORADO': 'CO',

        'CONNECTICUT': 'CT',

        'DISTRICT OF COLUMBIA': 'DC',

        'DELAWARE': 'DE',

        'FLORIDA': 'FL',

        'GEORGIA': 'GA',

        'HAWAII': 'HI',

        'IOWA': 'IA',

        'IDAHO': 'ID',

        'ILLINOIS': 'IL',

        'INDIANA': 'IN',

        'KANSAS': 'KS',

        'KENTUCKY': 'KY',

        'LOUISIANA': 'LA',

        'MASSACHUSETTS': 'MA',

        'MARYLAND': 'MD',

        'MAINE': 'ME',

        'MICHIGAN': 'MI',

        'MINNESOTA': 'MN',

        'MISSOURI': 'MO',

        'MISSISSIPPI': 'MS',

        'MONTANA': 'MT',

        'NORTH CAROLINA': 'NC',

        'NORTH DAKOTA': 'ND',

        'NEBRASKA': 'NE',

        'NEW HAMPSHIRE': 'NH',

        'NEW JERSEY': 'NJ',

        'NEW MEXICO': 'NM',

        'NEVADA': 'NV',

        'NEW YORK': 'NY',

        'OHIO': 'OH',

        'OKLAHOMA': 'OK',

        'OREGON': 'OR',

        'PENNSYLVANIA': 'PA',

        'RHODE ISLAND': 'RI',

        'SOUTH CAROLINA': 'SC',

        'SOUTH DAKOTA': 'SD',

        'TENNESSEE': 'TN',

        'TEXAS': 'TX',

        'UTAH': 'UT',

        'VIRGINIA': 'VA',

        'VERMONT': 'VT',

        'WASHINGTON': 'WA',

        'WISCONSIN': 'WI',

        'WEST VIRGINIA': 'WV',

        'WYOMING': 'WY',

        'PUERTO RICO': 'PR',

        'VIRGIN ISLANDS': 'VI'

     }

   

    return(state2abbrev)

 

 

 

##################################################################

### Create a Confusion Matrix

##################################################################

def make_confusion_matrix(cf,

                          group_names: list[str] = None,

                          categories: list[str] = 'auto',

                          count: bool = True,

                          percent: bool = True,

                          cbar: bool = True,

                          xyticks: bool = True,

                          xyplotlabels: bool = True,

                          sum_stats: bool = True,

                          figsize: tuple = None,

                          cmap: str = 'Blues',

                          title: str = None) -> None:

    '''

    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

   

    Parameters

    ---------

        - cf:            confusion matrix to be passed in

        - group_names:   List of strings that represent the labels row by row to be shown in each square.

        - categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

        - count:         If True, show the raw number in the confusion matrix. Default is True.

        - normalize:     If True, show the proportions for each category. Default is True.

        - cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.

                       Default is True.

        - xyticks:       If True, show x and y ticks. Default is True.

        - xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

        - sum_stats:     If True, display summary statistics below the figure. Default is True.

        - figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

        - cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'

                       See http://matplotlib.org/examples/color/colormaps_reference.html

 

        - title:         Title for the heatmap. Default is None.

    '''

 

    ## Other packages ##

    try:

        sys.path.append("/mnt/code/repairabletools/")

        import python

    except:

        from repairabletools import python

    try:

        import matplotlib.pyplot as plt

    except:

        python.install('matplotlib')

        import matplotlib.pyplot as plt 

    try: 

        import seaborn as sns

    except:

        python.install('seaborn')

        import seaborn as sns

   

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE

    blanks = ['' for i in range(cf.size)]

 

    if group_names and len(group_names)==cf.size:

        group_labels = ["{}\n".format(value) for value in group_names]

    else:

        group_labels = blanks

 

    if count:

        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]

    else:

        group_counts = blanks

 

    if percent:

        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]

    else:

        group_percentages = blanks

 

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]

    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

 

 

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS

    if sum_stats:

        #Accuracy is sum of diagonal divided by total observations

        accuracy  = np.trace(cf) / float(np.sum(cf))

 

        #if it is a binary confusion matrix, show some more stats

        if len(cf)==2:

            #Metrics for Binary Confusion Matrices

            precision = cf[1,1] / sum(cf[:,1])

            recall    = cf[1,1] / sum(cf[1,:])

            f1_score  = 2*precision*recall / (precision + recall)

            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(

                accuracy,precision,recall,f1_score)

        else:

            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)

    else:

        stats_text = ""

 

 

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS

    if figsize==None:

        #Get default figure size if not set

        figsize = plt.rcParams.get('figure.figsize')

 

    if xyticks==False:

        #Do not show categories if xyticks is False

        categories=False

 

 

    # MAKE THE HEATMAP VISUALIZATION

    plt.figure(figsize=figsize)

    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

 

    if xyplotlabels:

        plt.ylabel('True label')

        plt.xlabel('Predicted label' + stats_text)

    else:

        plt.xlabel(stats_text)

   

    if title:

        plt.title(title)   

 

       

        

##################################################################

###*********** The Following are for Pandas Dataframes ***********

################################################################## 

 

 

 

##################################################################

### Cleans Pandas Colummns

##################################################################

def clean_columns(df: pd.DataFrame(),

                 remove_timestamps: bool = False,

                 output_date_as_string: bool = False) -> pd.DataFrame:

    """

    This function will clean up and standardize column names and values in a Pandas dataframe. It is best to use this either right after reading in the data or before writing the data

   

    Parameters:

        - df (pd.DataFrame): DataFrame containing the state column.

        - (optional) remove_timestamp (bool): Removes timestamps from date columns (if True).

        - (optional) output_date_as_string: Outputs the date as a sting, as opposed to a date.

    """   

    ###Clean column names###

    #Get lowercase for all columns

    df.columns = [x.lower() for x in df.columns]

   

    #Remove unnamed columns

    df = df.loc[:, ~df.columns.str.contains('^unnamed')]

 

    #Replace spaces with special characters

    df.columns = df.columns.str.replace(' ', '_') #remove spaces ( ), replaces with underscore (_)

    df.columns = df.columns.str.replace('.', '_') #remove periods (.), replaces with underscore (_)

    df.columns = df.columns.str.replace('-', '_') #remove dashes (-), replaces with underscore (_)

    df.columns = df.columns.str.replace('$', '') #remove dollar signs ($)

    df.columns = df.columns.str.replace('%', 'pct') #remove percent signs (%), replaces with pct

    df.columns = df.columns.str.replace('#', 'num') #remove pound signs (#), replaces with num (number)

    df.columns = df.columns.str.replace(r'\(s\)', '', regex=True) #replaces (s) with s

    df.columns = df.columns.str.replace('(', '_') #Removes parentheses ((), replaces with underscore (_)

    df.columns = df.columns.str.replace(')', '_') #Removes parentheses ()), replaces with underscore (_)

    df.columns = df.columns.str.replace('+', '_') #Removes addition sign (+), replaces with underscore (_)

    df.columns = df.columns.str.replace('__', '_') #Removes double underscore (__), replaces with underscore (_)

    df.columns = df.columns.str.rstrip('_') #Removes trailing underscores (_)

    df.columns = df.columns.str.lstrip('_') #Removes leading underscores (_)

    df.columns = df.columns.str.replace('\W', '', regex=True) #Removes special characters

    df.columns = df.columns.str.strip() #Removes white space

    ## END ##

   

    ## Convert all values to uppercase ##

    #Object columns

    object_columns = df.select_dtypes(include=['object']).columns

    df[object_columns] = df[object_columns].apply(lambda x: x.astype(str).str.upper())

    ## END ##

   

                  

    ##Remove timestamp from date columns ##

    if remove_timestamps == True:

        #List of date columns

        date_cols = df.select_dtypes(include=['datetime64']).columns

 

        #Loop through each datetime

        for col in date_cols:

 

            #Extract only the date     

            df[col] = df[col].dt.date

 

            #Output the data as a string

            if output_date_as_string == False:

               

                #Convert back to datetime format

                df[col] = pd.to_datetime(df[col], errors='coerce')

 

    ## END ##        

        

                  

    ###Duplicate column names###

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()

 

    if duplicate_columns != []:

        logger.warning(f"Duplicate column names: {duplicate_columns}")

 

        #Remove duplicate column names

        df = df.loc[:,~df.columns.duplicated()]

                 

        logger.info('Removed duplicate columns')

    else:

        logger.info("No duplicate column names")

    ## END ##

   

    return(df)

 

 

       

##################################################################

### Match Month Name to Month Number

################################################################## 

def month_name_to_number(col: object) -> object:

    """

    This function will take the name of a month and output the associated number.

    Both 3 letter abbreviations of the month name as well as the full month name are acceptable.

   

    Parameters:

        - col = the Pandas dataframe column that you wish to change from month name to month number (e.g., df['month'])

    """

    

    d = {'JAN': 1,

         'FEB': 2,

         'MAR': 3,

         'APR': 4,

         'MAY': 5,

         'JUN': 6,

         'JUL': 7,

         'AUG': 8,

         'SEP': 9,

         'OCT': 10,

         'NOV': 11,

         'DEC': 12,

        }

   

    #Get first 3 letters of month

    col = col.str[:3].str.upper()

   

    try:

        #Make change

        col = col.map(d)

    except:

        raise Exception('ERROR: YOU HAVE ENTERED AN ERONEOUS MONTH NAME. PLEASE FIX.')

       

    return(col)

 

 

 

#Define function

def state_abbrev_mapping(col: object,

                         output_abbr: bool = False,

                         case: str = 'upper') -> object:

    """

    This function will take a Pandas dataframe column and change the from either the state name to its 2 letter abbreviation,

    or it will take the 2 letter abbreviation of the state and turn it into the full name of the state.

   

    Parameters:

        - col (object): Pandas dataframe column

        - (optional) output_abbr (bool): Output is a state abbreviation (as opposed to the state name)

        - (optional) case (str): The string case type (upper, lower, proper)

    """

        

    #Retrieve dictionary of states and their abbreviations

    state2abbrev = state_abbrev_dict()

 

    #List of states

    abbrev2state = {value: key for key, value in state2abbrev.items()}

   

            

    ## Map values ##

    #Make sure column is uppercase       

    col = col.str.upper().str.strip()

       

    #Is the output an abbreviation?

    #Apply changes

    if output_abbr == True:

        col = col.map(state2abbrev)

    else:

        col = col.map(abbrev2state)

    ## End ##

 

       

    ## Case (upper, lower, proper) ##       

    #Make string lowercase

    case = case.lower()

    #Remove case (e.g., lowercase) if applicable

    case = case.replace('case', '')

   

    #Does the user want a specific case sensitivity?

    if case == 'upper':

        col = col.str.upper()

    elif case == 'lower':

        col = col.str.lower()

    elif (case == 'proper') | (case == 'title'):

        col = col.str.title()

    else:

        raise Exception("ERROR: case IS INCORRECT. PLEASE PUT upper, lower, or proper!")

    ## End ##

           

    return(col)

 

 

 

##################################################################

### Get Difference of Data by Group and Date

##################################################################

def diff_by_date(df: pd.DataFrame,

                 group_col: str,

                 date_col: str,

                 col_list: list[str],

                 agg_type: str = 'mean',

                 fill_missing: bool = True) -> pd.DataFrame:

    """

    This function will take a Pandas dataframe and get the difference over time by category. For example, if you want the change in claim count by zip code over time, this function can provide that. Note that only numeric data (besides group_col and date_col) are kept. If you wish to keep categorical features, if appropriate, merge the orginal dataset with this one. Note that missing values are turned into 0s. The first date of each group will always be Null since there are no previous date to compare it to.

   

    

    Parameters:

        - df (pd.DataFrame): DataFrame containing the state column.

        - group_col (str): The column by which the data is grouped by (e.g., zip code, shop_id, etc.)

        - date_col (str): The date column.

        - col_list (list): A list of columns desired to get the change or difference over time.

        - (Optional) agg: How the data is aggregated (mean, sum, count, etc.)

        - (Optional) fill_missing: Boolean to fill in missing values for a given date and group with 0.

    """

       

    ## Column error testing ##

    error_test_string_col(df, group_col)

    error_col_in_df(df, date_col)

   

    for col in col_list:

        error_test_num_col(df, col)

    ## END ##

   

        

    ### Fill in missing data ###

    if fill_missing == True:

        group_codes = list(set(df[group_col]))

        dates = list(set(df[date_col]))

 

        #All combinations of groups and dates

        clean_data = pd.DataFrame(list(product(group_codes, dates)), columns=[group_col, date_col])

 

        #Fill in missing values

        df = pd.merge(clean_data, df,

                on = [group_col, date_col],

                how = 'left')

 

    #Sort data

    df = df.sort_values([group_col, date_col])

    ### END ###

   

    

    ### Difference from last entry ###

    #Reset value

    df_agg = pd.DataFrame()

 

    #Make sure aggregation is in lowercase

    agg_type = agg_type.lower()

   

    #Group by group and date

    if agg_type == 'mean':

        df_agg = df.sort_values([group_col, date_col], ascending = [True, False]).groupby([group_col, date_col]).mean(numeric_only=True)

    elif agg_type == 'sum':

        df_agg = df.sort_values([group_col, date_col], ascending = [True, False]).groupby([group_col, date_col]).sum(numeric_only=True)

    elif agg_type == 'count':

        df_agg = df.sort_values([group_col, date_col], ascending = [True, False]).groupby([group_col, date_col]).count(numeric_only=True)

 

    #Loop through columns

    for diff_col in col_list:

 

        #Make sure data is sorted correctly

        df_agg = df_agg.sort_values([group_col, date_col], ascending = [True, False])

       

        #Difference from current to previous date

        diff_df = pd.DataFrame(df_agg[diff_col] - df_agg.groupby(level=group_col)[diff_col].shift(-1))

 

        #New column name (diff = difference)

        new_col_name = diff_col + "_diff"

 

        #Rename column

        diff_df.rename(columns = {diff_col: new_col_name},inplace=True)

 

        #Combine data

        df_agg = pd.merge(df_agg, diff_df,

                left_index = True,

                right_index = True)

 

    #Reset index

    df_agg = df_agg.reset_index()

   

    return(df_agg)

 

 

 

##################################################################

### Aggregate data by n Rolling Months Within Group

##################################################################

def agg_rolling_intervals(df: pd.DataFrame,

                   date_col: str,

                   group_cols: list,

                   intervals: int,

                   interval_type: str = 'month',

                   agg_type: str = 'mean',

                   output_date_as_string: bool = False) -> pd.DataFrame:

    """

    This function will take a Pandas dataframe and gets a rolling sum or average by a specified group. For example, a rolling 7 day claim count by state.

   

    

    Parameters:

        - df (pd.DataFrame): DataFrame containing the state column.

        - date_col (str): The date column.

        - group_cols (list): A list of columns desired to be grouped by in the aggregation.

        - intervals (int): The number of intervals or periods that is rolling (e.g., 7 days, 3 months, etc.)

        - (optional) interval_type (str): The interval or period of time (month, year, or day).

        - (optional) agg_type (str): The aggregation of the data (mean or sum).

        - (optional) output_date_as_string (bool): Outputs the data as a string format as opposed to a datetime foramt.

    """

   

    #Copy the data

    df = df.copy()

   

    ## Get rolling n invervals ##

    if (interval_type == 'year') or (interval_type == 'years'):

       

        #Update day and month to first of the year

        df[date_col] = df[date_col].dt.to_period('Y').dt.to_timestamp()

       

        #Make sure name is consistent

        interval_type = 'year'

 

        #Get min and max dates

        min_date = df[date_col].min()

        max_date = df[date_col].max()

 

        #Yearly intervals (beginning of year)

        date_range = pd.date_range(start = min_date,

                 end = max_date, freq ='YS')

 

    elif (interval_type == 'month') or (interval_type == 'months'):

       

        #Update day of month

        df[date_col] = df[date_col].dt.to_period('M').dt.to_timestamp()

       

        #Make sure name is consistent

        interval_type = 'month'

       

        #Get min and max dates

        min_date = df[date_col].min()

        max_date = df[date_col].max()

 

        #Monthly intervals (beginning of month)

        date_range = pd.date_range(start = min_date,

                 end = max_date, freq ='MS')

 

    elif (interval_type == 'day') or (interval_type == 'days'):

       

        #Update day of month

        df[date_col] = df[date_col].dt.to_period('D').dt.to_timestamp()

       

        #Make sure name is consistent

        interval_type = 'day'

       

        #Get min and max dates

        min_date = df[date_col].min()

        max_date = df[date_col].max()

 

        #Daily intervales (beginning of day)

        date_range = pd.date_range(start = min_date,

                 end = max_date, freq ='D')

       

    ## END ##

   

 

    ## Aggregate Data ##

    #Initialize empty dataframe

    df_agg = pd.DataFrame()

 

    #Loop through all dates

    for date in date_range:

       

        #Mean of columns

        if agg_type == 'mean':

           

            #Get first of each month, year, or day

            if interval_type == 'month':

               

                #Subset dataframe

                subset = df.loc[(df[date_col] <= date) & (df[date_col] > (date - pd.DateOffset(months=intervals)))]

               

            elif interval_type == 'year':

               

                #Subset dataframe

                subset = df.loc[(df[date_col] <= date) & (df[date_col] > (date - pd.DateOffset(years=intervals)))]

               

            elif interval_type == 'day':

               

                #Subset dataframe

                subset = df.loc[(df[date_col] <= date) & (df[date_col] > (date - pd.DateOffset(days=intervals)))]

           

            

            #Get date

            subset.loc[:, date_col] = date

           

            #Group by date and group (mean)

            mean_df = subset.groupby(group_cols + [date_col]).mean(numeric_only=True).reset_index()

 

            #If grouping data

            if group_cols != []:

               

                #Group by date and group (sum), already filtered to correct date

                count_df = subset.groupby(group_cols).count().reset_index()

 

                #Rename column

                count_df = count_df.rename(columns = {date_col: 'count'})

 

                #Keep only

                count_df = count_df[group_cols + ['count']]

 

                #Aggregate data

                one_agg = pd.merge(mean_df, count_df,

                        on = group_cols,

                        how = 'left')

               

            #If no group

            else:

                one_agg = mean_df

                one_agg['count'] = subset.shape[0]  

        

        #Sum of columns

        if agg_type == 'sum':

           

            #Get first of each month, year, or day

            if interval_type == 'month':

               

                #Subset dataframe

                subset = df.loc[(df[date_col] <= date) & (df[date_col] > (date - pd.DateOffset(months=intervals)))]

               

            elif interval_type == 'year':

               

                #Subset dataframe

                subset = df.loc[(df[date_col] <= date) & (df[date_col] > (date - pd.DateOffset(years=intervals)))]

               

            elif interval_type == 'day':

               

                #Subset dataframe

                subset = df.loc[(df[date_col] <= date) & (df[date_col] > (date - pd.DateOffset(days=intervals)))]

           

            #Get date   

            subset.loc[:, date_col] = date

           

             #Group by date and group (sum)

            one_agg = subset.groupby(group_cols + [date_col]).sum(numeric_only=True).reset_index()

 

        #Concatenate data

        df_agg = pd.concat([df_agg, one_agg], axis = 0)

    ## END ##

    

    

    ## Format Data ##

    #Remove day or month if True

    if output_date_as_string == True:

        if interval_type == 'month':

            df_agg[date_col] = df_agg[date_col].apply(lambda x: x.replace(day=1)).dt.strftime('%m-%Y')

        elif interval_type == 'year':

            df_agg[date_col] = df_agg[date_col].apply(lambda x: x.replace(day=1)).dt.strftime('%Y')

    ## END ##

    return(df_agg)
