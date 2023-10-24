#!/usr/bin/env python
# coding: utf-8

# In[25]:


def calculate_specificity(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    specificity = tn / (tn + fp)
    return specificity


# In[26]:


import numpy as np

# Function to calculate confidence scores using bootstrapping
def calculate_confidence_score(y_true, y_pred_prob, num_bootstraps=1000, confidence_level=0.95):
    scores = []
    n = len(y_true)
    
    for _ in range(num_bootstraps):
        indices = np.random.choice(n, n, replace=True)
        y_true_bootstrap = y_true[indices]
        y_pred_prob_bootstrap = y_pred_prob[indices]
        
        fpr, tpr, _ = roc_curve(y_true_bootstrap, y_pred_prob_bootstrap)
        auc_score = auc(fpr, tpr)
        scores.append(auc_score)
    
    # Calculate the lower and upper percentiles for the confidence interval
    lower_percentile = ((1 - confidence_level) / 2) * 100
    upper_percentile = (confidence_level + (1 - confidence_level) / 2) * 100
    confidence_interval = np.percentile(scores, [lower_percentile, upper_percentile])
    
    return np.mean(scores), confidence_interval


# In[27]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb


# In[140]:


data = pd.read_csv("/Users/gustavosganzerla/Documents/multi_organisms/paper/review/data.csv")
data_temp = data[data['label']==1]
data_temp = data_temp.drop(columns = ["Unnamed: 0", "Unnamed: 0.1"])


# In[143]:


control_path = "/Users/gustavosganzerla/Documents/multi_organisms/paper/review/new_controls/physical_analysis/"
new_control = []

for item in os.listdir(control_path):
    if '.txt' in item:
        df = pd.read_csv(control_path+item, header= None, sep = "\t")
        new_control.append(df)
        
new_control = pd.concat(new_control, ignore_index=True)
new_control = new_control.dropna()
new_control[99] = new_control[99].astype(int)  
new_control.rename(columns={99: "label"}, inplace=True)


# In[147]:


data_temp.to_csv("/Users/gustavosganzerla/Documents/multi_organisms/paper/review/new_promoters.csv")
new_control.to_csv("/Users/gustavosganzerla/Documents/multi_organisms/paper/review/new_controls.csv")


# In[303]:


data = pd.read_csv("/Users/gustavosganzerla/Documents/multi_organisms/paper/review/new_data.csv")

X = data.iloc[:, 21:100].values
y = data.iloc[:, 100].values


# In[304]:


data


# In[153]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split your data into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel='linear', random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

# Train and evaluate each classifier
for classifier_name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    
    # Evaluate the classifier
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    print(f"{classifier_name} Evaluation:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("\n")


# In[154]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_classifier = xgb.XGBClassifier()
param_grid = {
    'n_estimators': [50, 100, 200], 
    'learning_rate': [0.01, 0.1, 0.2], 
    'max_depth': [3, 5, 7],             
}
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[287]:


seed = 42
np.random.seed(seed)

params = {
    'max_depth': 7,
    'learning_rate': 0.2,
    'n_estimators': 200,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Initialize K-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics for each fold
roc_auc_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []
specificity_scores = []

index = 1
# Perform 10-fold cross-validation
for train_index, test_index in kf.split(X, y):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize and fit the XGBoost classifier
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)
    
    # Predict probabilities
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Calculate evaluation metrics
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    specificity = calculate_specificity(confusion)
    
    # Append the scores to the lists
    roc_auc_scores.append(roc_auc)
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    specificity_scores.append(specificity)
    
    # Misclassification
    pd.DataFrame(y_pred).to_csv("~/Documents/multi_organisms/paper/review/missclassification/y_pred_fold_n"+str(index)+".csv")
    pd.DataFrame(y[test_index]).to_csv("~/Documents/multi_organisms/paper/review/missclassification/y_test_fold_n"+str(index)+".csv")
    pd.DataFrame(X[test_index]).to_csv("~/Documents/multi_organisms/paper/review/missclassification/X_test_fold_n"+str(index)+".csv")
    index+=1

# Display evaluation metrics for each fold
for i in range(10):
    print(f"Fold {i+1} - ROC AUC: {roc_auc_scores[i]:.2f}, Accuracy: {accuracy_scores[i]:.2f}, Precision: {precision_scores[i]:.2f}, Recall: {recall_scores[i]:.2f}, Specificity: {specificity_scores[i]:.2f}")

# Calculate and display mean and standard deviation of the evaluation metrics
print("\nMean Scores:")
print(f"Mean ROC AUC: {np.mean(roc_auc_scores):.2f}, Mean Accuracy: {np.mean(accuracy_scores):.2f}, Mean Precision: {np.mean(precision_scores):.2f}, Mean Recall: {np.mean(recall_scores):.2f}, Mean Specificity: {np.mean(specificity_scores):.2f}")
print("Standard Deviations:")
print(f"Std. ROC AUC: {np.std(roc_auc_scores):.2f}, Std. Accuracy: {np.std(accuracy_scores):.2f}, Std. Precision: {np.std(precision_scores):.2f}, Std. Recall: {np.std(recall_scores):.2f}, Std. Specificity: {np.std(specificity_scores):.2f}")


# In[234]:


# Initialize a plot for ROC curves
plt.figure(figsize=(8, 6))

# Initialize lists to store ROC AUC scores and confidence scores for each fold
roc_auc_scores = []
confidence_scores = []

# Perform 10-fold cross-validation
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize and fit the XGBoost classifier
    clf_1 = xgb.XGBClassifier(**params)
    clf_1.fit(X_train, y_train)
    
    # Predict probabilities
    y_pred_prob = clf_1.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Append the ROC AUC score to the list
    roc_auc_scores.append(roc_auc)
    
    # Calculate confidence score using bootstrapping
    auc_mean, auc_confidence_interval = calculate_confidence_score(y_test, y_pred_prob)
    
    # Append the confidence score to the list
    confidence_scores.append(auc_confidence_interval)
    
    # Plot ROC curve for each fold with AUC and confidence score
    plt.plot(fpr, tpr, lw=2, label=f'Fold {len(roc_auc_scores)} (AUC = {roc_auc:.2f}, Confidence = [{auc_confidence_interval[0]:.2f}, {auc_confidence_interval[1]:.2f}])')

# Plot the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set plot labels and legend
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Show the plot
plt.show()


# In[288]:


####misclassification v.2
import os

folder = "/Users/gustavosganzerla/Documents/multi_organisms/paper/review/missclassification/"

X = pd.DataFrame()

for item in os.listdir(folder):
    if '.csv' in item and 'X_test' in item:
        temp_df = pd.read_csv(folder+item)
        X = X.append(temp_df, ignore_index=True)

X = pd.DataFrame(X)

y_test = pd.DataFrame()

for item in os.listdir(folder):
    if '.csv' in item and 'y_test' in item:
        temp_df = pd.read_csv(folder+item)
        y_test = y_test.append(temp_df, ignore_index=True)

y_test = pd.DataFrame(y_test)

y_pred = pd.DataFrame()

for item in os.listdir(folder):
    if '.csv' in item and 'y_pred' in item:
        temp_df = pd.read_csv(folder+item)
        y_pred = y_pred.append(temp_df, ignore_index=True)

y_pred = pd.DataFrame(y_pred)

y_pred = y_pred.rename(columns={"0":"predicted"})
y_test = y_test.rename(columns={"0":"actual"})
       
combined_df = pd.concat([X, y_test, y_pred], axis=1)
        
combined_df = combined_df.drop(columns = ['Unnamed: 0'])


# In[289]:


tp = combined_df[(combined_df['actual'] == 1) & (combined_df['predicted'] == 1)]
tn = combined_df[(combined_df['actual'] == 0) & (combined_df['predicted'] == 0)]
fp = combined_df[(combined_df['actual'] == 0) & (combined_df['predicted'] == 1)]
fn = combined_df[(combined_df['actual'] == 1) & (combined_df['predicted'] == 0)]


# In[291]:


fn


# In[292]:


fn_sampled = fn.sample(n=1810)
missclassified_df = fp.append(fn_sampled, ignore_index=True)

X = missclassified_df.iloc[:, 0:79].values
y = missclassified_df.iloc[:, 79].values


# In[293]:


#fp = 1810
#fn = 2406
missclassified_df


# In[294]:


mean_tp = tp.mean().values
mean_tn = tn.mean().values
mean_fp = fp.mean().values
mean_fn = fn.mean().values

mean_tp = mean_tp[:-2]
mean_tn = mean_tn[:-2]
mean_fp = mean_fp[:-2]
mean_fn = mean_fn[:-2]


# In[295]:


# Plot the lines for each data array

plt.figure(figsize=(10,6))

x = np.arange(-59, 20)

# Plot the lines and markers for each data array
plt.plot(x, mean_tp, label='True Positives', marker='o', linestyle='-')
plt.plot(x, mean_tn, label='True Negatives', marker='s', linestyle='--')
plt.plot(x, mean_fp, label='False Positives', marker='^', linestyle='-.')
plt.plot(x, mean_fn, label='False Negatives', marker='d', linestyle=':')

# Add labels, title, and legend
plt.xlabel('Nucleotide Position')
plt.ylabel('Kcal/mol^bp-1')
plt.legend()

# Show the plot
plt.grid(True)  # Add grid lines
plt.show()


# In[296]:


import scipy.stats as stats

statistic, p_value = stats.shapiro(df)
p_value


# In[297]:


import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Create a DataFrame from the data
df = pd.DataFrame({'TP': mean_tp, 'TN': mean_tn, 'FP': mean_fp, 'FN': mean_fn})

# Create a DataFrame
df = pd.DataFrame(df)

# Reshape the data for Tukey HSD
data_melted = pd.melt(df, var_name='Group', value_name='Value')

# Perform Kruskal-Wallis test
h_statistic, p_value = kruskal(df['TP'], df['TN'], df['FP'], df['FN'])
print('Kruskal-Wallis p-value:', p_value)

# Perform Tukey HSD post hoc test
tukey_results = pairwise_tukeyhsd(data_melted['Value'], data_melted['Group'], alpha=0.05)

# Display the Tukey results
print(tukey_results)

# Create boxplots
plt.figure(figsize=(10, 6))
df.boxplot()
plt.ylabel('Kcal/mol^bp-1')
plt.grid(False)

# Show the boxplots
plt.show()


# In[194]:


promoters = data[data['label']==1]


mean_promoters = promoters.mean().values
mean_promoters = mean_promoters[:-1]


# In[196]:


mean_promoters = mean_promoters[21:100]


# In[202]:


plt.figure(figsize=(10,6))

x = np.arange(-59, 20)

# Plot the lines and markers for each data array
plt.plot(x, mean_promoters, label='Promoters', marker='o', linestyle='-')

# Add labels, title, and legend
plt.xlabel('Nucleotide Position')
plt.ylabel('Kcal/mol^bp-1')
plt.legend()

# Show the plot
plt.grid(True)  # Add grid lines
plt.show()


# In[298]:


fn_sampled = fn.sample(n=1810)
missclassified_df = fp.append(fn_sampled, ignore_index=True)

X = missclassified_df.iloc[:, 0:79].values
y = missclassified_df.iloc[:, 79].values


# In[299]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split your data into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel='linear', random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

# Train and evaluate each classifier
for classifier_name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    
    # Evaluate the classifier
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    print(f"{classifier_name} Evaluation:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("\n")


# In[247]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_classifier = xgb.XGBClassifier()
param_grid = {
    'n_estimators': [50, 100, 200], 
    'learning_rate': [0.01, 0.1, 0.2], 
    'max_depth': [3, 5, 7],             
}
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[251]:


seed = 42
np.random.seed(seed)

# Initialize K-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics for each fold
roc_auc_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []
specificity_scores = []

index = 1
# Perform 10-fold cross-validation
for train_index, test_index in kf.split(X, y):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize and fit the XGBoost classifier
    clf_2 = xgb.XGBClassifier()
    clf_2.fit(X_train, y_train)
    
    # Predict probabilities
    y_pred_prob = clf_2.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Calculate evaluation metrics
    y_pred = clf_2.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    specificity = calculate_specificity(confusion)
    
    # Append the scores to the lists
    roc_auc_scores.append(roc_auc)
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    specificity_scores.append(specificity)
    

# Display evaluation metrics for each fold
for i in range(10):
    print(f"Fold {i+1} - ROC AUC: {roc_auc_scores[i]:.2f}, Accuracy: {accuracy_scores[i]:.2f}, Precision: {precision_scores[i]:.2f}, Recall: {recall_scores[i]:.2f}, Specificity: {specificity_scores[i]:.2f}")

# Calculate and display mean and standard deviation of the evaluation metrics
print("\nMean Scores:")
print(f"Mean ROC AUC: {np.mean(roc_auc_scores):.2f}, Mean Accuracy: {np.mean(accuracy_scores):.2f}, Mean Precision: {np.mean(precision_scores):.2f}, Mean Recall: {np.mean(recall_scores):.2f}, Mean Specificity: {np.mean(specificity_scores):.2f}")
print("Standard Deviations:")
print(f"Std. ROC AUC: {np.std(roc_auc_scores):.2f}, Std. Accuracy: {np.std(accuracy_scores):.2f}, Std. Precision: {np.std(precision_scores):.2f}, Std. Recall: {np.std(recall_scores):.2f}, Std. Specificity: {np.std(specificity_scores):.2f}")


# In[250]:


# Initialize a plot for ROC curves
plt.figure(figsize=(8, 6))

# Initialize lists to store ROC AUC scores and confidence scores for each fold
roc_auc_scores = []
confidence_scores = []

# Perform 10-fold cross-validation
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize and fit the XGBoost classifier
    clf_2 = xgb.XGBClassifier(**params)
    clf_2.fit(X_train, y_train)
    
    # Predict probabilities
    y_pred_prob = clf_2.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Append the ROC AUC score to the list
    roc_auc_scores.append(roc_auc)
    
    # Calculate confidence score using bootstrapping
    auc_mean, auc_confidence_interval = calculate_confidence_score(y_test, y_pred_prob)
    
    # Append the confidence score to the list
    confidence_scores.append(auc_confidence_interval)
    
    # Plot ROC curve for each fold with AUC and confidence score
    plt.plot(fpr, tpr, lw=2, label=f'Fold {len(roc_auc_scores)} (AUC = {roc_auc:.2f}, Confidence = [{auc_confidence_interval[0]:.2f}, {auc_confidence_interval[1]:.2f}])')

# Plot the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set plot labels and legend
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Show the plot
plt.show()


# In[300]:


validation = pd.read_csv("/Users/gustavosganzerla/Documents/multi_organisms/bacteria/classif/datasets_to_classify/validation/line_stability_all_promoters_regulondb.txt",
                       header = None, sep = "\t")

X = validation.iloc[:, 1:80].values



# In[253]:


df_ecoli_validation = pd.DataFrame()
df_ecoli_validation['xgb1'] = clf_1.predict(X)
df_ecoli_validation['xgb2'] = clf_2.predict(X)


# In[301]:


#df_ecoli_validation[df_ecoli_validation['xgb1']==0]
df_ecoli_validation[(df_ecoli_validation['xgb1'] == 1) | (df_ecoli_validation['xgb2']==1)]


# In[308]:


from tqdm import tqdm

# Set input and output paths
path_validation = "/Volumes/sd/physical_analysis/"
path_validation_out = "/Volumes/sd/lists_of_promoters/"

# Load validation files and make predictions
results = []
for filename in tqdm(os.listdir(path_validation)):
    if filename.endswith(".txt"):
        
        # Load validation data
        validation = pd.read_csv(os.path.join(path_validation, filename), sep="\t", header=None)
        X_validation = validation.iloc[:, 1:80].values
    


# In[377]:


from tqdm import tqdm

path_validation = "/Users/gustavosganzerla/Documents/multi_organisms/bacteria/physical_analysis_ecoli/"
path_validation_out = "/Volumes/sd/lists_of_promoters_ecoli/"

for filename in tqdm(os.listdir(path_validation)):
    if filename.endswith(".txt"):
        validation = pd.read_csv(os.path.join(path_validation, filename), sep="\t", header=None)
        
        if not validation.empty:
            X_validation = validation.iloc[:, 1:80].values
            validation['xgb1'] = clf_1.predict(X_validation)
            validation['xgb2'] = clf_2.predict(X_validation)
            validation['xgb1_proba'] = clf_1.predict_proba(X_validation)[:,1]
            validation['xgb2_proba'] = clf_2.predict_proba(X_validation)[:,1]
            
            filtered = validation[(validation['xgb1'] == 1) | (validation['xgb2'] == 1)]
            
             # Extract the first two columns as a NumPy array
            values = filtered.iloc[:, [0, -2, -1]].values.tolist()

            # Remove square brackets, commas, and quotes and format the values
            formatted_values = [list(map(str, row)) for row in values]
            
            # Write values to a CSV file
            output_filename = os.path.splitext(filename)[0] + "_filtered.csv"
            output_path = os.path.join(path_validation_out, output_filename)
            
            with open(output_path, 'w') as csv_file:
                for row in formatted_values:
                    csv_file.write(','.join(row) + '\n')
        else:
            print(f"The validation dataframe for {filename} is empty.")
            


# In[380]:


import os
from tqdm import tqdm
import pandas as pd
import re
import csv
# Set input and output paths
path_promoters = "/Volumes/sd/lists_of_promoters_ecoli/"
path_physical = "/Users/gustavosganzerla/Documents/multi_organisms/bacteria/physical_analysis_ecoli/"
path_annotation = "/Volumes/sd/reference_annotation_ecoli/"
path_upstream = "/Volumes/sd/e_coli_upstream/"
path_output = "/Users/gustavosganzerla/Documents/multi_organisms/paper/review/predictions_ecoli/"


# Initialize a dictionary to store IDs and their corresponding prediction values
id_predictions = {}

for predicted_promoter_fname in os.listdir(path_promoters):
    if predicted_promoter_fname.endswith('.csv'):
        filename_physical = predicted_promoter_fname.replace("_filtered.csv", ".txt")
        filename_annotation = predicted_promoter_fname.replace("_filtered.csv", ".ft")
        filename_annotation2 = predicted_promoter_fname.replace("_upstream_filtered.csv", "_feature.tab")

        # Initialize a dictionary to store prediction values for the current file
        id_predictions_current = {}

        # Load predicted promoter IDs and their prediction values into the dictionary
        with open(os.path.join(path_promoters, predicted_promoter_fname), 'r') as f_promoters:
            reader = csv.reader(f_promoters)
            for row in reader:
                if len(row) >= 3:
                    # The ID is in the first column
                    promoter_id = row[0]
                    # The prediction values are in the second and third columns
                    xgb1_pred = row[1]
                    xgb2_pred = row[2]

                    # Store the ID and prediction values in the dictionary
                    id_predictions_current[promoter_id] = [xgb1_pred, xgb2_pred]

        # Load upstream sequences into a dictionary
        upstream_sequences = {}
        with open(os.path.join(path_upstream, filename_annotation), 'r') as f_upstream:
            reader = csv.reader(f_upstream, delimiter='\t')
            for row in reader:
                if len(row) == 10:
                    upstream_sequences[row[2]] = row[6]

        # Process annotation file
        with open(os.path.join(path_annotation, filename_annotation2), 'r') as f_annotation, \
             open(os.path.join(path_output, filename_physical), 'a+') as f_output:
            reader = csv.reader(f_annotation, delimiter='\t')
            f_output.write("Column 1: NCBI ID"+"\n"+"Column 2: Organism name"+"\n"+
                           "Column 3: RSAT ID"+"\n"+"Column 4: Start Position"+"\n"+
                           "Column 5: End position"+"\n"+"Column 6: Forward (F) or Reverse (R)"+"\n"+
                           "Column 7: Classification score"+"\n"+"Column 8: Computationally predicted promoter label"+"\n"
                           "Column 9: Promoter sequence (-60 to +1)"+"\n"+"Column 10: Functional annotation of the gene"+"\n\n\n"
                           
            )
            for row in reader:
                if len(row) == 11 and row[0] in id_predictions_current:
                    # Look up upstream sequence in dictionary
                    sequence = upstream_sequences.get(row[0], '')

                    annot = 'Derived by automated computational analysis using gene prediction method: '
                    if annot in row[7]:
                        function = re.sub(annot, "", row[7])
                    else:
                        function = row[7]

                    if 'Protein Homology.;' in function:
                        function = function.replace("Protein Homology.;", "")
                    if 'GeneMarkS+.;' in function:
                        function = function.replace('GeneMarkS+.;', "")
                    if 'frameshifted;' in function:
                        function = function.replace('frameshifted;', "")

                    # Compute start position
                    if row[5] != '<NA>':
                        start = int(row[5]) - 80
                    else:
                        start = 0  # or some other default value

                    # Write output
                    name = re.sub("_upstream.ft", "", filename_annotation)

                    # Get the prediction values for the current ID
                    prediction_values = id_predictions_current[row[0]]
                    pred = round(float(max(prediction_values)),2)
                    
                    ai_label = "Computationally predicted promoter"
                    
                    if row[6]=="D":
                        row[6]=="F"
                    
                    f_output.write('\t'.join([
                        row[3], name, row[0], str(start), row[5], row[6], str(pred), 
                        ai_label, sequence[340:400], str(function), 
                        ]) + '\n')


# In[388]:


###saving models

filename1 = "/Users/gustavosganzerla/Documents/multi_organisms/paper/review/xgboost_1.model"
clf_1.get_booster().save_model(filename1)

filename2 = "/Users/gustavosganzerla/Documents/multi_organisms/paper/review/xgboost_2.model"
clf_2.get_booster().save_model(filename2)


# In[387]:


###TAXON ID

path = "/Users/gustavosganzerla/Documents/multi_organisms/paper/review/predictions/"

out = open("/Users/gustavosganzerla/Documents/multi_organisms/paper/review/taxons.txt", "a+")
list_names = []

for item in os.listdir(path):
    aux = item.split("_GCF")
    name = aux[0].replace("_", " ")
    out.write(name+"\n")
    
    


# In[386]:


len(list_names)

