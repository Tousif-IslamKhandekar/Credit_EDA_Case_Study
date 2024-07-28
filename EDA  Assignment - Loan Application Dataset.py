#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('application_data.csv')


# In[3]:


pd.set_option('display.max_columns', None)


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.columns


# In[7]:


df.info()


# In[8]:


df.shape


# ## Checking for data imbalance

# In[9]:


(100*df['TARGET'].value_counts(normalize=True)).plot(kind="pie",autopct='%1.1f%%')
plt.show()


# **At the outset we need to keep in mind that TARGET variable data is imbalanced or skewed as approx 92% of the data belongs to non-default**

# In[10]:


df['CODE_GENDER'].value_counts().plot.bar()
plt.show()


# **Number of males is half the number of Females in the data. Hence we can say data is skewed or imbalanced towards females**

# In[11]:


df.head()


# **We can see from the head that there are null values present in some of the columns. Let's go ahead and do some more analysis on null values.**

# In[12]:


pd.set_option('display.max_rows', None)
df.isnull().sum()


# **We see a worrying number of null values in some of the columns. Let's go ahead and group the columns with a criteria.**

# In[13]:


drop_cols = []
for i in df.columns:
    if (df[i].isnull().sum() / len(df[i]) * 100) > 50:
        drop_cols.append(i)


# In[14]:


drop_cols


# ### As suspected, since the above columns have more than 50% null values, I would prefer to drop the columns from the dataset.

# In[15]:


df.drop(drop_cols,axis=1,inplace=True)


# In[16]:


df.shape


# In[17]:


df.head()


# In[18]:


df.drop(['FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21'],axis=1,inplace=True)


# **Flag document columns are dropped because the purpose and nature of the columns is unclear even in the data dictionary. Hence in my opinion they do not have any impact in the analysis.**

# In[19]:


df.shape


# ### Categorical Numerical Analysis

# **Gender vs Income**

# In[20]:


df.groupby('CODE_GENDER')['AMT_INCOME_TOTAL'].mean()


# In[131]:


df.groupby('CODE_GENDER')['AMT_INCOME_TOTAL'].mean().plot.bar()
plt.ylabel('AMT_INCOME_TOTAL')
plt.show()


# ### Note : mean income is greater for men than in case of females.

# In[22]:


df.head()


# **Standardizing Data**

# In[23]:


df[['DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']] = df[['DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']].abs()


# In[24]:


df['DAYS_BIRTH'] = (df.DAYS_BIRTH / 365).abs()


# In[25]:


df.rename(columns={'DAYS_BIRTH' : 'DAYS_BIRTH_YEARS'}, inplace=True)


# In[26]:


df[['DAYS_EMPLOYED','DAYS_REGISTRATION']] = df[['DAYS_EMPLOYED','DAYS_REGISTRATION']].abs()


# In[27]:


df.head()


# In[28]:


df.drop(['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START'],axis=1,inplace=True)


# In[29]:


df.shape


# **I have dropped the above columns because I think which Day or Hour of the application will not have any impact on the analysis of loan default**

# **For better convenience in the analysis let's segment the columns into numeric and categorical columns.**

# In[134]:


num_col = []
cat_col = []
extra_col= []


# In[135]:


for i in df.columns:
    if df[i].nunique()>30:
        num_col.append(i)
    elif df[i].nunique()<30:
        cat_col.append(i)
    else:
        extra_col.append(i)


# In[136]:


print(len(num_col))
print(len(cat_col))
print(len(extra_col))


# **For better and convenient analysis of numerical and categorical variables, I have divided the columns into groups. The criteria/threshold taken is 30 unique categories.Although I have taken 30 as the criteria here, let me give heads up here, this is not a hard and fast rule, criteria may vary and depends on the sense of the analyst. Even in this analysis going further we may need to change or make an exception in the criteria, if needed.** 

# ## Categorical Univariate/Bivariate Analysis

# In[139]:


for i in cat_col:
    plt.figure(figsize=[10,4])
    plt.subplot(1, 2, 1)
    df[i].value_counts(normalize=True).plot.bar()
    plt.xlabel('VALUE_COUNTS')
    plt.subplot(1,2,2)
    df.groupby(i)['TARGET'].mean().plot.bar()
    plt.ylabel('Mean_TARGET')
    plt.show()


# **In the above analysis, We are analyzing the value counts of the categorical variables and in the second plot we are comparing them with the mean of the TARGET variable, to analyze the influence of the respective categorical variable on the default rate.**

# **Interesting Observations:**

# - Default rate of Males is higher than females, although the fact that males have a higher income than females, which is established in the Gender vs Income analysis.
# 
# - People who don't own their car have a higher rate of default. Although point to be noted here is, number of records of people who don't own car are considerably higher than people who do. Hence, there can be a data imbalance here. But I think we can take a benefit of doubt here.
# 
# - People who don't own realty have a higher default rate. There is a clear indication here as the records of people who own realty are much higher than people who don't.
# 
# - People with 9 and 11 children have an unusual high default rate. But I assume, that number of records for these cases will be very low, hence we don't have a large enough sample size to get a sureshot analysis.
# 
# - Unemployed and Maternity Leave people have a higher rate of default. Although the records are very less for these categories compared to others. Hence we can't establish the analysis here.
# 
# - People with Lower secondary and secondary education have higher rates of default.
# 
# - People with civil marriages and Single people have a clear higher default rates.
# 
# - People with rented apartments and people who are living with parents are more likely to default.
# 
# - Clients who have not provided their home phone have higher default rate than those who have.
# 
# - Low-skill labourers have highest rate of default. Second highest are Drivers.
# 
# - Clients who live in region rated 3 have highest default rate, followed by 2 and then 1. This signifies region with 3 rating are red flags for the approver. Same observation holds true for city rating as well.
# 
# - People whose regional permanent and contact addresses are different are more likely to default.
# 
# - People whose permanent region do not match work regions are more likely to default.
# 
# - People whose regional contact address do not match regional work address are more likely to default.
# 
# - People whose contact city do not match permanent city are more likely to default.
# 
# - People whose permanent city do not match work city are more likely to default.
# 
# - People whose contact city does not match work city are more likely to default.
# 
# - Clients whose living conditions is rated under emergency state mode are more likely to default.
# 
# - WRT observations in client's surroundings who have definitely defaulted, we see a sharp increase in default rate as the number of social surrounding defaulting goes up. This gives a hint about the client if he's more likely to default by looking at his social group.
# 
# - WRT number of enquiries to credit bureau before applcation whether hour,day,week,month,quarter or year before application, we see the same trend. As the number of enquiries increase the chance of defaulting also increase.
# 
# 
# 
# 

# In[34]:


df.isnull().sum() / df.shape[0] * 100


# **We still see some very high percentage of null values in the dataset. This might affect the analysis which we don't want. Hence let's drop the columns with more than 40% of null values.**

# In[35]:


more_drop_cols = []
for i in df.columns:
    if (df[i].isnull().sum() / len(df[i]) * 100) > 40:
        more_drop_cols.append(i)


# In[36]:


more_drop_cols


# In[37]:


df.drop(more_drop_cols,axis=1,inplace=True)


# In[38]:


df.shape


# In[39]:


for i in more_drop_cols:
    if i in num_col:
        num_col.remove(i)
    elif i in cat_col:
        cat_col.remove(i)
    else:
        extra_col.remove(i)


# In[40]:


print(len(num_col))
print(len(cat_col))
print(len(extra_col))


# In[41]:


num_col


# In[42]:


df.ORGANIZATION_TYPE.nunique()


# **We see that unexpectedly ORGANIZATION_TYPE column has popped up in num_col with 58 categories. Ideally it should be in cat_col list. We will have to make an exception here and append this column to cat_col and drop it from num_col.**

# In[43]:


num_col.remove('ORGANIZATION_TYPE')


# In[44]:


cat_col.append('ORGANIZATION_TYPE')


# **Organization_Type vs Default Rate**

# In[45]:


plt.figure(figsize=[20,10])
plt.subplot(1, 2, 1)
df['ORGANIZATION_TYPE'].value_counts(normalize=True).plot.barh()
plt.subplot(1,2,2)
df.groupby('ORGANIZATION_TYPE')['TARGET'].mean().plot.barh()
plt.show()
    


# **We can see Transport Industry people have a very higher rate of default**

# ## Numerical Univariate Analysis

# In[46]:


for i in num_col:
    plt.figure(figsize=[8,5])
    plt.subplot(1,2,1)
    df[i].plot.hist(x=df[i],bins=50)
    plt.xlabel(i)
    plt.subplot(1,2,2)
    df[i].plot.box()
    plt.show()


# **We see a large amount of outliers in many of the variable distributions in the box plots plotted. Will do further analysis on this.**

# **Handling Outliers and Standardizing Values**

# In[47]:


df.AMT_INCOME_TOTAL.quantile([0.25,0.50,0.75,0.90,0.95,0.99,1])


# **Here we see that the difference between 99 percentile and 100 percentile is quite a lot but the progression from 25th percentile to 99th percentile is normal. Hence we can cap the income @ 99th percentile in order to ignore the outliers.**

# In[48]:


df = df[df.AMT_INCOME_TOTAL<=472500]  ##capping income


# In[49]:


df.AMT_INCOME_TOTAL.plot.box()
plt.show()


# **Now we see a better distribution of income with no outliers.**

# **Doing the same for Goods_price, Annuity and Credit variables.**

# In[50]:


df.AMT_GOODS_PRICE.quantile([0.25,0.50,0.75,0.90,0.95,0.99,1])


# In[51]:


df = df[df.AMT_GOODS_PRICE<=1800000]  ##capping goods_price


# In[52]:


df.AMT_ANNUITY.quantile([0.25,0.50,0.75,0.90,0.95,0.99,1])


# In[53]:


df = df[df.AMT_ANNUITY<=68512]  ##capping Annuity


# In[54]:


df.AMT_CREDIT.quantile([0.25,0.50,0.75,0.90,0.95,0.99,1])


# **Distribution of Credit looks fine with minimum outliers. Hence no need to cap Credit.**

# ## Numerical Categorical Analysis

# In[55]:


for i in num_col:
    plt.figure(figsize=[10,4])
    plt.subplot(1,2,1)
    df[i].plot.box(x=df[i])
    plt.subplot(1,2,2)
    df.groupby("TARGET")[i].median().plot.bar()
    plt.show()


# **Observations:**

# - Clients who have defaulted have a lower median income.
# - Median credit amount of loans that have defaulted is slightly lower.
# - Annuity amount of default loans is slightly higher.
# - Median age of defaulters is lower than non-defaulters.
# - Defaulters have started their current job more recently than non-defaulters.
# - Defaulters have changed their registration more recently than non-defaulters.
# - Defaulters have changed their identification more recently than non-defaulters.
# - External Source rating is less for defaulters.
# - Defaulters have changed their phone more recently than non-defaulters.

# **We have to again make an exception here, and move OBS_30_CNT_SOCIAL_CIRCLE and OBS_60_CNT_SOCIAL_CIRCLE variables to categorical columns group, even though these variables have more than 30 categories.**

# ### Segmented Univariate Analysis

# **We can bin the income column into income groups**

# In[56]:


df['AMT_INCOME_TOTAL_GROUP'] = pd.cut(df.AMT_INCOME_TOTAL,[0,50000,100000,200000,300000,400000,500000],labels=["<50k","50k-1L","1L-2L","2L-3L", "3L-4L",'4L-5L'])


# In[57]:


df.AMT_INCOME_TOTAL_GROUP.value_counts(normalize= True)


# In[140]:


plt.figure(figsize=[10,4])
plt.subplot(1,2,1)
df.AMT_INCOME_TOTAL_GROUP.value_counts(normalize=True).plot.bar()
plt.subplot(1,2,2)
df.groupby('AMT_INCOME_TOTAL_GROUP')['TARGET'].mean().plot.bar()
plt.ylabel('Mean_TARGET')
plt.show()


# **We can observe that default rate is highest in case of 1L-2L bucket, second highest in 50k-1L bucket. Third highest is less than 50000 bucket.**

# **Doing the same binning for other columns**

# In[59]:


df['AMT_CREDIT_GROUP'] = pd.cut(df.AMT_CREDIT,[0,500000,1000000,1500000,2000000],labels=['<5L','5L-10L','10L-15L','15L-20L'])


# In[60]:


df.AMT_CREDIT_GROUP.value_counts(normalize= True)


# In[141]:


plt.figure(figsize=[10,4])
plt.subplot(1,2,1)
df.AMT_CREDIT_GROUP.value_counts(normalize=True).plot.bar()
plt.subplot(1,2,2)
df.groupby('AMT_CREDIT_GROUP')['TARGET'].mean().plot.bar()
plt.ylabel('MEAN_TARGET')
plt.show()


# **Highest default rate is for 5L-10L bucket, second highest is for below 5L bucket.**

# In[62]:


df['AMT_ANNUITY_GROUP'] = pd.cut(df.AMT_ANNUITY,[0,10000,20000,30000,40000,50000,60000,70000],labels=["<10K","10k-20k","20k-30k","30k-40k", "40k-50k","50k-60k","60k-70k"])


# In[63]:


df.groupby('AMT_ANNUITY_GROUP')['TARGET'].mean().plot.bar()
plt.show()


# **Highest rate of default is for 30k-40k bucket.**

# ### Numerical Bivariate Analysis

# In[64]:


for col1 in num_col:
    for col2 in num_col:
        if (col1 != col2) & (col1 != 'SK_ID_CURR') & (col2 != 'SK_ID_CURR'):
            sns.scatterplot(x=df[col1],y=df[col2])
            plt.show()


# **Observations:**

# *Correlation & Causation*

# - Credit amount and Annuity have a positive linear relationship, which is obvious because if the credit amount of the loan is more then installment will also be more.
# - Credit amount of the loan increases with Goods price. This is also understandable because for higher goods prices more credit will be given by bank.
# - Annuity and Goods_price also have a linear positive relationship because for higher goods price higher loans are given for which higher annuity is given. Hence causation in this case is the loan credit and not goods_price.
# - Days Registration has a somewhat linear positive relation with age. This is because more the age, the more likely it is that the registration was changed more earlier. However it is difficult to establish causation here.
# - ID change time increases with age which is understandale because less the age it is likely that the client has changed identification more recently.
# - Observed defaulters in client's surroundings increases linearly for 30 and 60 days, trend can be clear after observing for 30 days which increases in 60 days.

# In[65]:


sns.pairplot(df[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']])
plt.show()


# **We can observe the linear relationships as we have observed in the bivariate analysis above.**

# ## Numerical vs Categorical Bivariate Analysis

# In[66]:


for col1 in cat_col:
    for col2 in num_col:
        print('Boxplot of',col1,'vs',col2)
        sns.boxplot(x=df[col1],y=df[col2])
        plt.xticks(rotation=90)
        plt.show()


# **Observations:**

# - Median age of defaulters is less as compared to that of non-defaulters.
# - Median external rating is less for defaulters.
# - Median Credit amount for revolving loans is less.
# - Median Annuity is less revolving loans.
# - Goods_price is less in case of revolving loans.
# - Median income of females is less than that of males.
# - Median age and overall age distribution of females is more than males.
# - Median income who own their own car is more.
# - Those who own their own car have higher credit amount of loans.
# - Median age of people who dont own their own car is more.
# - Median age of people who own their own realty is more.
# - Median income of Businessman is highest followed by Commercial associate.
# - Credit amounts for unemployed and maternity leave clients is highest.
# - External rating of student and pensioner is higher and lowest for unemployed.
# - Median income of lower secondary educated is the lowest and highest for academic degree.
# - Median age of the lower secondary educated people is higher.
# - Managers have highest median income.
# - Managers have highest credit amount in loans.
# - Clients with highest region rating of 1, have highest median income and loan credits.
# - Clients with highest city rating of 1, have highest median income and higher
# - Clients with same permanent and contact address are older.
# - Clients whose permanent address do not match address work address have higher income and lesser median age.
# - Clients whose contact address do not match work address have higher median income and lesser median age.
# - Same above observations are seen at city level as well.
# 
# 
#   

# In[67]:


sns.boxplot(data=df,x=df['AMT_INCOME_TOTAL'],y=df['NAME_INCOME_TYPE'])
plt.show()


# In[68]:


sns.boxplot(data=df,x=df['AMT_CREDIT'],y=df['NAME_INCOME_TYPE'])
plt.show()


# In[69]:


sns.boxplot(data=df,x=df['AMT_ANNUITY'],y=df['NAME_INCOME_TYPE'])
plt.show()


# **Let's load the previous_application data and check the dataset.**

# In[70]:


df1 = pd.read_csv('previous_application.csv')


# In[71]:


df1.head()


# In[72]:


df1.shape


# In[73]:


df1.info()


# In[74]:


df1.isnull().sum() / df1.shape[0] * 100


# **We will be proceeding with the same approach as we have done with the application dataset. We will be dropping the columns with more than 40% null values.**

# In[75]:


cols_to_drop = []
for i in df1.columns:
    if (df1[i].isnull().sum() / df1.shape[0] * 100) > 40:
        cols_to_drop.append(i)
        
    


# In[76]:


cols_to_drop


# In[77]:


df1.drop(cols_to_drop,axis=1,inplace=True)


# In[78]:


df1.shape


# In[79]:


df1.head()


# In[80]:


df1.drop(['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY'],axis=1,inplace=True)


# In[81]:


df1.drop(['SK_ID_PREV'],axis=1,inplace=True)


# **I have dropped the above columns as I think the above columns will have no impact on the analysis.**

# In[82]:


df1.shape


# In[83]:


df1.NAME_CASH_LOAN_PURPOSE.unique()


# In[84]:


df1['DAYS_DECISION'] = df1.DAYS_DECISION.abs()


# In[85]:


df1.CODE_REJECT_REASON.unique()


# In[86]:


df1.head()


# In[87]:


df1.drop(['SELLERPLACE_AREA'],axis=1,inplace=True)


# **I decided to drop the above column because the meaning of this column is unclear to me and hence I do not want to use this column in the analysis.**

# **Looks like CNT_PAYMENT column is in months. To standardize the data let's convert it into years.**

# In[88]:


df1.CNT_PAYMENT = df1.CNT_PAYMENT / 12


# In[89]:


set(list(df.columns)).intersection(set(list(df1.columns)))


# **Good to know for analysis that these columns are present in both the datasets.**

# In[90]:


df1.shape


# In[91]:


df1_num_col = []
df1_cat_col = []
df1_extra_col = []


# In[92]:


for i in df1.columns:
    if df1[i].nunique()>30:
        df1_num_col.append(i)
    elif df1[i].nunique()<30:
        df1_cat_col.append(i)
    else:
        df1_extra_col.append(i)


# In[93]:


print(len(df1_num_col))
print(len(df1_cat_col))
print(len(df1_extra_col))


# In[94]:


print(df1_num_col)


# In[95]:


for i in df1_num_col:
    df1[i] = pd.to_numeric(df1[i], errors='coerce')


# In[96]:


for i in df1_num_col:
    df1.groupby('NAME_CONTRACT_STATUS')[i].mean().plot.bar()
    plt.ylabel(i)
    plt.show()


# **Observations:**

# - Cancelled loans by customers have highest Annuity amount.
# - Refused loans have highest loan amount asked for.
# - Credit amount is highest for refused loans.
# - Canceled loans have highest Goods_price amount.
# - Approved loans have longest gap between current and previous applications. This is obvious because if the loans were approved the client wouldn't have required another loan for a long time.
# - Canceled loans have highest credit term.

# In[97]:


df1_cat_col


# ## Categorical Categorical Analysis

# In[98]:


for i in df1_cat_col:
    plt.figure(figsize=[10,8])
    df1.groupby('NAME_CONTRACT_STATUS')[i].value_counts(normalize=True).unstack().plot.bar()
    plt.ylabel(i)
    lgd = plt.legend(loc=9, bbox_to_anchor=(1.05, 1.1))
    plt.show()


# **Observations:**

# - Consumer loans have highest rate of approval.
# - Repeat customers have a higher chance of getting their loans approved or cancelling the loan during approval process or not taking the loan after it is offered.
# - POS portfolio have highest rate of approved loans.
# - Customers acquired country-wide have a higher chance of getting loans approved and customers acquired through credit and cash offices have a higher chance of cancelling the loan.
# - Loans with mid-range Interest rates have highest approval rate.Loans with low-normal interest rates have high refusal rate.
# - POS Household with Interest have highest approval rate.

# ## Merging the DataFrames

# **As we can see the common column in the dataframes. Let's merge the dataframes on this column.**

# In[99]:


df_new = df.merge(df1,on='SK_ID_CURR',how='inner')


# In[100]:


df_new.head()


# In[101]:


df_new.drop(['SK_ID_CURR'],axis=1,inplace=True)


# **We no longer need the ID column after merging the dataframes.**

# In[102]:


df_new.shape


# In[103]:


df_new.info()


# **Let's regroup the columns into numeric and categorical in the combined dataframe.**

# In[104]:


num_col = []
cat_col = []
extra_col = []


# In[105]:


for i in df_new.columns:
    if df_new[i].nunique()>30:
        num_col.append(i)
    elif df_new[i].nunique()<30:
        cat_col.append(i)
    else:
        extra_col.append(i)


# In[106]:


print(len(num_col))
print(len(cat_col))
print(len(extra_col))


# ## Numerical Bivariate Analysis

# In[107]:


num_col


# In[108]:


df_new.DAYS_ID_PUBLISH = df_new.DAYS_ID_PUBLISH.astype(int)


# **TARGET vs Categorical variable of previous_application data.**

# In[109]:


for i in df_new[['AMT_ANNUITY_y','AMT_APPLICATION','AMT_CREDIT_y','AMT_GOODS_PRICE_y','DAYS_DECISION','CNT_PAYMENT']]:
                        df_new.groupby('TARGET')[i].mean().plot.bar()
                        plt.ylabel(i)
                        plt.show()


# **Observations:**

# - Clients who have defaulted have less annuity amount.
# - Clients who have defaulted asked for lesser credit amount in their previous applications.
# - Defaulters have a higher goods_price amount.
# - Decisions made on previous applications are more recent than non-defaulters.
# - Previous credit term of the applications is lesser for non-defaulters. Defaulters had a higher credit term.

# In[110]:


cat_col


# **Numerical Variables of previous_application vs TARGET.**

# In[111]:


for i in df_new[['NAME_CONTRACT_TYPE_y','NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE','CODE_REJECT_REASON','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO','NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP','PRODUCT_COMBINATION']]:
    df_new.groupby(i)['TARGET'].mean().plot.bar()
    plt.show()


# **Observations.**

# - Revolving loans had a higher default rate in the previous applications as well.
# - Under Loan Purpose - 'Refusal to name the goal' and 'Hobby' have highest default rates.
# - New Client has highest default rate.
# - Insurance category for goods have highest default rate followed by Vehicles.
# - Cards as a porfolio has highest default rate.
# - walk-in customers have higher default rate.
# - AP Cash Loan acquisition channel have highest default rate followed by Contact center.
# - Auto Technology channel have highest default rate.
# - High Interest rates loans have higher default rate.
# - cash street and middle combination have highest default rate. Second highest is Card street.

# In[112]:


df_new['NAME_CONTRACT_STATUS'].value_counts(normalize=True)*100


# ## Numerical vs Categorical Analysis of the combined Dataframe

# In[113]:


num_col.remove('ORGANIZATION_TYPE')


# In[114]:


cat_col.append('ORGANIZATION_TYPE')


# In[115]:


num_col


# In[116]:


df1_cat = ['NAME_CONTRACT_TYPE_y','NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE','CODE_REJECT_REASON','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO','NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP','PRODUCT_COMBINATION']


# ## Numerical vs Categorical Bivariate Analysis

# In[117]:


for col1 in df1_cat:
    for col2 in num_col:
            print('Boxplot of',col1,'vs',col2)
            sns.boxplot(x=df_new[col1],y=df_new[col2])
            plt.xticks(rotation=90)
            plt.show()
                               


# **Observations:**

# - Applicants who have taken business loans fall under highest income group.
# - Applicants who have taken car loans have one of the highest credit amount for loans.
# - Clients who have taken loans for medicine and water supply have a higher median age.
# - Customers with higher median age have a higher chance of cencelling loans.
# - Clients with approved loans have a higher external ratings.
# - Approved loans have longer gaps in between current and previous application.
# - Cancelled loans have higher credit term. Loans with higher credit terms have a higher chance of cancellation by customers.
# - Cashless payment types have higher external ratings.
# - Refreshed clients have higher external ratings.
# - Car loans have higher credit amount, annuity and goods price.
# - Clients who have taken cash loans have higher median age.
# - Customers who were cross-selled loans have a higher median age. Hence cross-selling has a better chance of working on older customers.
# - Car dealers have a higher income, annuity, credit amount and goods price.
# - Car dealers have a higher external rating.
# - Tourism industry sellers have a higher median income.
# - MLM and Construction industry have a higher median age.
# - High income individuals go for Cash Street and low interest combination.
# - Low interest rate and cross-selling combo has a good success rate with older people.
# 

# ## Multivariate Analysis

# **Creating Heatmap:**

# **Let's create a heatmap to visualize the correlation between variables in the combined dataset.**

# In[129]:


plt.figure(figsize=[20,10])
sns.heatmap(df_new[num_col].corr(),annot=True,cmap="RdYlGn")
plt.show()


# ## The above heatmap clearly shows the correlation between the numerical columns in the combined dataframe.

# In[ ]:




