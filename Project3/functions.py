#Eliminate duplicates
def drop_dup_df(data):
    return data.drop_duplicates()

#Null percentages
def calculate_null_percentage(data):
    nulls=(data.isnull().sum() / len(data)) * 100
    return nulls.round(1).sort_values(ascending=False)

#Log Transformations
def log_transfom_clean_(x):   
    if np.isfinite(x) and x!=0:
        return np.log(x)
    else:
        return np.NAN 


#Square root transformations
def sqrt_transfom_clean_(x): 
    if np.isfinite(x) and x>=0:
        return np.sqrt(x)
    else:
        return np.NAN


#Eliminate categorical variables that start with "frequency"
def remove_the_frequency_columns(dataframe, prefix):
    columns_to_remove=[col for col in dataframe.columns if col.lower().startswith(prefix.lower())]
    return dataframe.drop(columns=columns_to_remove)

#Histograms
def histograms_plot(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=10,ax=ax,facecolor='blue')
        ax.set_title(feature+" distribution",color='black')
    fig.tight_layout()  
    plt.show()
