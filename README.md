Key takeaways
===============================================

This repository is a collection of Datacamp's projects



A Network Analysis of Game of Thrones
----------------------------------


### Tags

### Codes


Original creator: []()

A New Era of Data Analysis in Baseball
----------------------------------


### Tags

### Codes


Original creator: []()

A Visual History of Nobel Prize Winners
----------------------------------


### Tags

### Codes


Original creator: []()

Analyze Your Runkeeper Fitness Data
----------------------------------


### Tags

### Codes


Original creator: []()

Analyzing TV Data
----------------------------------


### Tags

### Codes


Original creator: []()

ASL Recognition with Deep Learning
----------------------------------

American Sign Language (ASL) is the primary language used by many deaf individuals in North America, and it is also used by hard-of-hearing and hearing individuals. The language is as rich as spoken languages and employs signs made with the hand, along with facial gestures and bodily postures.

In this project, you will train a convolutional neural network to classify images of ASL letters. After loading, examining, and preprocessing the data, you will train the network and test its performance.

### Tags

`Data Manipulation` `Data Visualization` `Machine Learning` `Importing & Cleaning Data`

### Codes

- Determine the global random seed with Tensorflow
- One-Hot encoder

```python
import tensorflow as tf
from keras.utils import np_utils

# Sets the global random seed 
tf.set_random_seed(2)

# One-hot encode the training labels
# Start from zero(0)
y_train_OH = np_utils.to_categorical(y_train, num_classes=3)
```

Original creator: [Alexis Cook](https://www.datacamp.com/instructors/alexiscook)

Bad Passwords and the NIST Guidelines
----------------------------------


### Tags

### Codes


Original creator: []()

Book Recommendations from Charles Darwin
----------------------------------


### Tags

### Codes


Original creator: []()

Classify Song Genres from Audio Data
----------------------------------

Using a dataset comprised of songs of two music genres (Hip-Hop and Rock), you will train a classifier to distinguish between the two genres based only on track information derived from Echonest (now part of Spotify). You will first make use of `pandas` and `seaborn` packages in Python for subsetting the data, aggregating information, and creating plots when exploring the data for obvious trends or factors you should be aware of when doing machine learning.

Next, you will use the `scikit-learn` package to predict whether you can correctly classify a song's genre based on features such as danceability, energy, acousticness, tempo, etc. You will go over implementations of common algorithms such as PCA, logistic regression, decision trees, and so forth.

### Tags

`Data Manipulation` `Data Visualization` `Machine Learning` `Importing & Cleaning Data`

### Codes

- If there is no obvious elbow on a scree plot of variance amount in data, we use cumulative explained variance instead and determine the threshold such as `0.85` (rule of thumb)

```python
# Merge the relevant columns of tracks and echonest_metrics
echo_tracks = echonest_metrics.merge(tracks[['genre_top', 'track_id']], on='track_id')

from sklearn.decomposition import PCA

# Get our explained variance ratios from PCA using all features
pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_

# plot the explained variance using a barplot
fig, ax = plt.subplots()
ax.bar(range(pca.n_components_), exp_variance)
ax.set_xlabel('Principal Component #')

```

Original creator: [Ahmed Hasan](https://www.datacamp.com/instructors/ahmedhasan)

Comparing Cosmetics by Ingredients
----------------------------------


### Tags

### Codes


Original creator: []()

Comparing Search Interest with Google Trends
----------------------------------

Time series data is everywhere; from temperature records, to unemployment rates, to the S&P 500 Index. Another rich source of time series data is Google Trends, where you can freely download the search interest of terms and topics from as far back as 2004. This project dives into manipulating and visualizing Google Trends data to find unique insights.

In this project’s guided variant, you will explore the search data underneath the Kardashian family's fame and make custom plots to find how the most famous Kardashian/Jenner sister has changed over time. In the unguided variant, you will analyze the worldwide search interest of five major internet browsers to calculate metrics such as rolling average and percentage change.

### Tags

`Data Manipulation` `Data Visualization` `Importing & Cleaning Data`

### Codes

- Cast the columns to `int` type
- Cast the columns to type `datetime64[ns]`
- Smooth out the fluctuations with rolling means
- 
```python
# Cast the columns to integer type
trends[col] = pd.to_numeric(trends[col])
# Cast the columns to type `datetime64[ns]`
trends['month'] = pd.to_datetime(trends['month'])
# Smooth out the fluctuations with rolling means
trends.rolling(window=12).mean()
```

Original creator: [David Venturi](https://www.datacamp.com/instructors/davidventuri)

Disney Movies and Box Office Success
----------------------------------


### Tags

### Codes


Original creator: []()

Do Left-handed People Really Die Young?
----------------------------------


### Tags

### Codes


Original creator: []()

Dr. Semmelweis and the Discovery of Handwashing
----------------------------------

In 1847, the Hungarian physician Ignaz Semmelweis made a breakthough discovery: he discovers handwashing. Contaminated hands was a major cause of childbed fever and by enforcing handwashing at his hospital he saved hundreds of lives.

### Tags

`Data Manipulation` `Data Visualization` `Probability & Statistics` `Importing & Cleaning Data` `Data Manipulation` `Data Visualization` `Case Studies`

### Codes

- Bootstrap analysis

```python
# A bootstrap analysis of the reduction of deaths due to handwashing
boot_mean_diff = []
for i in range(3000):
    boot_before = before_washing['proportion_deaths'].sample(frac=1, replace=True)
    boot_after = after_washing['proportion_deaths'].sample(frac=1, replace=True)
    boot_mean_diff.append(boot_before.mean() - boot_after.mean())

# Calculating a 95% confidence interval from boot_mean_diff 
confidence_interval = pd.Series(boot_mean_diff).quantile([0.025, 0.975])
confidence_interval
```

Original creator: [Rasmus Bååth](https://www.datacamp.com/instructors/rasmus-baath)

Exploring the Bitcoin Cryptocurrency Market
----------------------------------


### Tags

### Codes


Original creator: []()

Exploring the Evolution of Linux
----------------------------------


### Tags

### Codes


Original creator: []()

Exploring the History of Lego
----------------------------------

The [Rebrickable](https://rebrickable.com/downloads/) database includes data on every LEGO set that has ever been sold; the names of the sets, what bricks they contain, what color the bricks are, etc. It might be small bricks, but this is big data! In this project, you will get to explore the Rebrickable database and answer a series of questions related to the history of Lego!

### Tags

`Data Manipulation` `Data Visualization` `Importing & Cleaning Data`

### Codes

- Extract information from Pandas's `groupby` 
- `.size` includes `NaN` values while `count` does not
```python
import pandas 

# colors_summary: Distribution of colors based on transparency
colors_summary = colors.groupby('is_trans').count()
nan_colors_summary = colors.groupby('is_trans').size()

# Create a summary of average number of parts by year: `parts_by_year`
parts_by_year = sets.groupby('year')['num_parts'].mean()
# themes_by_year: Number of themes shipped by year
themes_by_year = sets.groupby('year')[['theme_id']].nunique()
```

Original creator: [Ramnath Vaidyanathan](https://www.datacamp.com/instructors/ramnath)

Exploring the NYC Airbnb Market
----------------------------------


### Tags

### Codes


Original creator: []()

Extract Stock Sentiment from News Headlines
----------------------------------


### Tags

### Codes


Original creator: []()

Find Movie Similarity from Plot Summaries
----------------------------------


### Tags

### Codes


Original creator: []()

Generating Keywords for Google Ads
----------------------------------


### Tags

### Codes


Original creator: []()

Give Life: Predict Blood Donations
----------------------------------


### Tags

### Codes


Original creator: []()

Investigating Netflix Movies and Guest Stars in The Office
----------------------------------
In this project, you’ll apply the skills you learned in Introduction to Python and Intermediate Python to solve a real-world data science problem. You’ll press “watch next episode” to discover if Netflix’s movies are getting shorter over time and which guest stars appear in the most popular episode of "The Office", using everything from lists and loops to pandas and matplotlib.

You’ll also gain experience in an essential data science skill — exploratory data analysis. This will allow you to perform critical tasks such as manipulating raw data and drawing conclusions from plots you create of the data.

### Tags

`Data Manipulation` `Data Visualization` `Programming`

### Codes

- Give colors according to genres
- Reverse boolean mask `~`
```python
# Define an empty list
colors = []

# Iterate over rows of netflix_movies_col_subset
for ind_movie, movie in netflix_movies_col_subset.iterrows():
    if movie['genre'] == 'Children':
        colors.append('red')
    elif movie['genre'] == 'Documentaries':
        colors.append('blue')
    elif movie['genre'] == 'Stand-Up':
        colors.append('green')
    else:
        colors.append('black')

# Set the figure style and initalize a new figure
plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(12,8))

# Create a scatter plot of duration versus release_year
plt.scatter(netflix_movies_col_subset['release_year'], netflix_movies_col_subset['duration'], c=colors)

# Create a title and axis labels
plt.title('Movie duration by year of release')
plt.xlabel('Release year')
plt.ylabel('Duration (min)')

# Show the plot
plt.show()

# Reverse boolean mask
typical_netflix_movies = netflix_movies_col_subset[~netflix_movies_col_subset['genre'].isin(['Children', 
                                                                                             'Documentaries', 
                                                                                             'Stand-Up'])]
typical_netflix_movies.head()
```

Original creator: [Justin Saddlemyer](https://www.datacamp.com/instructors/justin-saddlemyer)

Mobile Games A/B Testing with Cookie Cats
----------------------------------


### Tags

### Codes


Original creator: []()

Naïve Bees: Deep Learning with Images
----------------------------------


### Tags

### Codes


Original creator: []()

Naïve Bees: Image Loading and Processing
----------------------------------
Can a machine distinguish between a honey bee and a bumble bee? Being able to identify bee species from images, while challenging, would allow researchers to more quickly and effectively collect field data. In this Project, you will use the Python image library Pillow to load and manipulate image data. You'll learn common transformations of images and how to build them into a pipeline.

This project is the first part of a series of projects that walk through working with image data, building classifiers using traditional techniques, and leveraging the power of deep learning for computer vision. The second project in the series is Naïve Bees: Predict Species from Images.

### Tags

`Data Manipulation` `Data Visualization` `Machine Learning` `Importing & Cleaning Data`

### Codes

- `PIL` library for images

```python
from PIL import Image

plt.imshow(img_data)
plt.imshow(img_data[:,:, 0], cmap=plt.cm.Reds_r)

honey = Image.open('datasets/bee_12.jpg')

# convert honey to grayscale
honey_bw = honey.convert('L')
display(honey_bw)

# convert the image to a NumPy array
honey_bw_arr = np.array(honey_bw)

# get the shape of the resulting array
honey_bw_arr_shape = honey_bw_arr.shape
print("Our NumPy array has the shape: {}".format(honey_bw_arr_shape))

# save a image
honey_bw.save("saved_images/bw_flipped.jpg")

# plot the array using matplotlib
plt.imshow(honey_bw_arr, cmap=plt.cm.gray)
plt.show()

# plot the kde of the new black and white array
plot_kde(honey_bw_arr, 'k')
```

Original creator: [Peter Bull](https://www.datacamp.com/instructors/peter-929a90ab-0e62-41e0-af37-39066fab8e60), [Emily Miller](https://www.datacamp.com/instructors/emily-b513c106-1c3d-4b38-a0a5-443773993c6d)

Naïve Bees: Predict Species from Images
----------------------------------


### Tags

### Codes


Original creator: []()

Name Game: Gender Prediction using Sound
----------------------------------


### Tags

### Codes


Original creator: []()

Predicting Credit Card Approvals
----------------------------------

Commercial banks receive a lot of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!). Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays. In this project, you will build an automatic credit card approval predictor using machine learning techniques, just like the real banks do.

The dataset used in this project is the [Credit Card Approval dataset](http://archive.ics.uci.edu/ml/datasets/credit+approval) from the UCI Machine Learning Repository.

### Tags

`Data Manipulation` `Machine Learning` `Importing & Cleaning Data` `Applied Finance`

### Codes

- Read `.data` file by Pandas's `read_csv()`
- Impute missing values by mean imputation
- Impute with the most frequent value 
- Convert the non-numeric data into numeric
- GridSearchCV

```python
# Read .data file
cc_apps = pd.read_csv('datasets/cc_approvals.data', header=None)

# Impute the missing values with mean imputation
cc_apps_train.fillna(cc_apps_train.mean(axis=0), inplace=True)
cc_apps_test.fillna(cc_apps_train.mean(axis=0), inplace=True)

# Count the number of NaNs in the datasets and print the counts to verify
print(cc_apps_train.isnull().sum())
print(cc_apps_test.isnull().sum())

# Iterate over each column of cc_apps_train
for col in cc_apps_train:
    # Check if the column is of object type
    if cc_apps_train[col].dtypes == 'object':
        # Impute with the most frequent value
        cc_apps_train = cc_apps_train.fillna(cc_apps_train[col].value_counts().index[0])
        cc_apps_test = cc_apps_test.fillna(cc_apps_test[col].value_counts().index[0])

# Convert the categorical features in the train and test sets to numeric data independently
cc_apps_train = pd.get_dummies(cc_apps_train)
cc_apps_test = pd.get_dummies(cc_apps_test)

from sklearn.model_selection import GridSearchCV

# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)
```

Original creator: [Sayak Paul](https://www.datacamp.com/instructors/spsayakpaul)

Real-time Insights from Social Media Data
----------------------------------


### Tags

### Codes


Original creator: []()

Recreating John Snow's Ghost Map
----------------------------------


### Tags

### Codes


Original creator: []()

Reducing Traffic Mortality in the USA
----------------------------------


### Tags

### Codes


Original creator: []()

Risk and Returns: The Sharpe Ratio
----------------------------------

When you assess whether to invest in an asset, you want to look not only at how much money you could make but also at how much risk you are taking. The Sharpe Ratio, developed by Nobel Prize winner William Sharpe some 50 years ago, does precisely this: it compares the return of an investment to that of an alternative and relates the relative return to the risk of the investment, measured by the standard deviation of returns.

In this project, you will apply the Sharpe ratio to real financial data using pandas.

### Tags

`Applied Finance` `Case Studies`

### Codes

- Extract information in finance data

```python
# calculate daily stock_data returns
# Pandas method
stock_returns = stock_data.pct_change()

# calculate the daily sharpe ratio
# divide avg_excess_return by sd_excess_return
daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)

# annualize the sharpe ratio
# multiply daily_sharpe_ratio by 
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)
```

Original creator: [Stefan Jansen](https://www.datacamp.com/instructors/stefanc7679de853c74cadb413cd65b3c3dd74)

Streamlining Employee Data
----------------------------------


### Tags

### Codes


Original creator: []()

The Android App Market on Google Play
----------------------------------


### Tags

### Codes


Original creator: []()

The GitHub History of the Scala Language
----------------------------------


### Tags

### Codes


Original creator: []()

The Hottest Topics in Machine Learning
----------------------------------
Neural Information Processing Systems (NIPS) is one of the top machine learning conferences in the world where groundbreaking work is published. In this Project, you will analyze a large collection of NIPS research papers from the past decade to discover the latest trends in machine learning. The techniques used here to handle large amounts of data can be applied to other text datasets as well.

Familiarity with Python and pandas is required to complete this Project, as well as experience with Natural Language Processing in Python (`sklearn` specifically).

### Tags

`Data Manipulation` `Data Visualization` `Machine Learning` `Probability & Statistics` `Importing & Cleaning Data`

### Codes

- `map` and `lambda`
- `wordcloud` library for NLP
- NLP in scikit-learn

```python
# Import the wordcloud library
import wordcloud 

# Join the different processed titles together.
long_string = ' '.join(papers['title_processed'])

# Create a WordCloud object
wordcloud = wordcloud.WordCloud()

# Generate a word cloud
# -- YOUR CODE HERE --
wordcloud.generate(long_string)

# Visualize the word cloud
wordcloud.to_image()

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(papers['title_processed'])

# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)

# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
```

Original creator: [Lars Hulstaert](https://www.datacamp.com/instructors/larshulstaert)

What and Where are the World's Oldest Businesses
----------------------------------


### Tags

### Codes


Original creator: []()

Which Debts Are Worth the Bank's Effort?
----------------------------------


### Tags

### Codes


Original creator: []()

Who Is Drunk and When in Ames, Iowa?
----------------------------------


### Tags

### Codes


Original creator: []()

Who's Tweeting? Trump or Trudeau?
----------------------------------


### Tags

### Codes


Original creator: []()

Word Frequency in Classic Novels
----------------------------------


### Tags

### Codes


Original creator: []()

Writing Functions for Product Analysis
----------------------------------


### Tags

### Codes


Original creator: []()

