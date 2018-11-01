"""
Capstone Project

“Date-A-Scientist”

Machine Learning Fundamentals
Jeroen van Renen
Slack: @jpvrenen
cohort-sep-18-2018


Question:
Can we predict if someone has children or not?

Columns to use:
age
drugs
smokes
education
religion

Using:
KNN
K-Means


"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

#Create your df here:
profiles = pd.read_csv('profiles.csv')

#Size of the dataframe
print(profiles.shape)
#What are the columns
print(profiles.columns)

#Perhaps we can take a look at age and income?
plt.scatter(profiles["age"], profiles["income"], alpha=0.1)
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Age vs Income")
plt.show()

#Take a look at income
print(profiles.income.value_counts())
plt.scatter(profiles["age"],profiles["has_kid"], alpha=0.1)
plt.show()

#Create mappings for 'age, drinks, drugs, smokes, religion'
kids_mapping = {"doesn&rsquo;t have kids": 0, "doesn&rsquo;t have kids, but might want them": 0,\
               "doesn&rsquo;t have kids, but wants them": 0, "doesn&rsquo;t want kids": 0,\
               "has kids": 1, "has a kid": 1, "doesn&rsquo;t have kids, and doesn&rsquo;t want any": 0,\
               "has kids, but doesn&rsquo;t want more": 1, "has a kid, but doesn&rsquo;t want more": 1,\
               "has a kid, and might want more": 1, "wants kids": 0, "might want kids": 0,\
               "has kids, and might want more": 1, "has a kid, and wants more": 1,\
               "has kids, and wants more": 1}

religion = ["other", "agnosticism", "agnosticism but not too serious about it",\
            "agnosticism and laughing about it", "catholicism but not too serious about it",\
            "atheism", "other and laughing about it", "atheism and laughing about it", "christianity",\
            "christianity but not too serious about it", "other but not too serious about it",\
            "judaism but not too serious about it", "atheism but not too serious about it",\
            "catholicism", "christianity and somewhat serious about it", "atheism and somewhat serious about it",\
            "other and somewhat serious about it", "catholicism and laughing about it",\
            "judaism and laughing about it", "buddhism but not too serious about it",\
            "agnosticism and somewhat serious about it", "judaism", "christianity and very serious about it",\
            "atheism and very serious about it", "catholicism and somewhat serious about it",\
            "other and very serious about it", "buddhism and laughing about it", "buddhism",\
            "christianity and laughing about it", "buddhism and somewhat serious about it",\
            "agnosticism and very serious about it", "judaism and somewhat serious about it",\
            "hinduism but not too serious about it", "hinduism", "catholicism and very serious about it",\
            "buddhism and very serious about it", "hinduism and somewhat serious about it", "islam",\
            "hinduism and laughing about it", "islam but not too serious about it", "islam and somewhat serious about it",\
            "judaism and very serious about it", "islam and laughing about it", "hinduism and very serious about it",\
            "islam and very serious about it"]

religion_mapping = dict()
for i in range(len(religion)):
    religion_mapping[religion[i]] = i

drinks_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}

#Map the data
profiles["has_kid"] = profiles.offspring.map(kids_mapping)
profiles["religion_code"] = profiles.religion.map(religion_mapping)
profiles["drinks_code"] = profiles.drinks.map(drinks_mapping)
profiles["drugs_code"] = profiles.drugs.map(drugs_mapping)
profiles["smokes_code"] = profiles.smokes.map(smokes_mapping)


#Create feature data including labels
feature_data_labels = profiles[['smokes_code', 'drinks_code', 'drugs_code', 'religion_code', 'has_kid']]
#Let's drop all NaN's from our 'feature_data_labels'
feature_data_labels.dropna(inplace=True)
#Create labels
feature_labels = feature_data_labels['has_kid']
#Create features
feature_data = feature_data_labels[['smokes_code', 'drinks_code', 'drugs_code', 'religion_code']]

#normalize the data
x = feature_data.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

#Finally create train/test sets
train_data, test_data, train_labels, test_labels = train_test_split(feature_data, feature_labels, test_size = 0.2, random_state = 1)
classifier = KNeighborsClassifier(n_neighbors = 59)
classifier.fit(train_data, train_labels)


#Show score for different 'k', lets see how accurate this model is!
scores = []
for k in range(1, 200):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(train_data, train_labels)
    scores.append(classifier.score(test_data, test_labels))
    
plt.plot(range(1,200), scores)
plt.show()

#Hang on this might be accurate but how predictive is this model?
predictions = classifier.predict(test_data)
print(test_labels.value_counts())


##K-Means
k = 2
# Use KMeans() to create a model that finds 2 clusters
model = KMeans(n_clusters=k)

# Use .fit() to fit the model to samples
model.fit(feature_data)
# Use .predict() to determine the labels of samples 
labels = model.predict(feature_data)
# Print the labels
print(labels)

# Code starts here:
itemsize_labels=len(feature_labels)
offspring = np.chararray(feature_labels.shape, itemsize=itemsize_labels)

for i in range(len(feature_data)):
  if feature_labels.iloc[i] == 0:
    offspring[i] = 'no kid'
  elif feature_labels.iloc[i] == 1:
    offspring[i] = 'kid'

df = pd.DataFrame({'kmeans_labels': labels , 'offspring': offspring})
print(df)

ct = pd.crosstab(df['kmeans_labels'], df['offspring'])
print(ct)

