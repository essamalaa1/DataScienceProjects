# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression , Lasso
from sklearn.metrics import mean_squared_error, r2_score
from numpy.linalg import norm
import joblib


# %%
data = pd.read_csv(r"C:\Users\Essam\Desktop\Assignment 1 ML\loan_old.csv")
data.dropna(inplace=True)
data = data.drop(["Loan_ID"],axis=1)

# %%
data.head(10)

# %%
data.info()

# %%
data.isnull().sum()

# %%
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    mean = data[column].mean()
    std = data[column].std()
    print(f"Column: {column}")
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")
    print("------------")


# %% [markdown]
# ### * As shown above , we have missing values
# 

# %% [markdown]
# ## check for numerical and categorical values

# %%
nf = data.select_dtypes(include=['int', 'float'])
cf= data.select_dtypes(include=['object'])
print("numerical features: ")
print(nf.columns)
print("categorical features: ")
print(cf.columns)

# %% [markdown]
# ### Check if the numerical data is at the same scale or not 

# %% [markdown]
# #### * looks like the some of the  numerical features has high standard deviation thus not at same scale

# %%
sns.pairplot(nf)
plt.show()

# %% [markdown]
# ### Records containing missing values are removed
# 

# %%
data = data.dropna()

# %%
data.isnull().sum()

# %%
# corr=data.corr()
# corr
# sns.heatmap(corr,annot=True)

# %% [markdown]
# #### Split the data , the features and targets are separated for logistic regresion
# 

# %%
X = data.drop(columns=["Max_Loan_Amount","Loan_Status"], axis=1)
y = data["Max_Loan_Amount"]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

# %% [markdown]
# ### Encode the categorical attributes

# %%
columns_to_encode = ["Gender", "Married", "Dependents", "Education", "Loan_Tenor", "Credit_History", "Property_Area"]

for column in columns_to_encode:
    encoder = LabelEncoder()
    X_train[column] = encoder.fit_transform(X_train[column])
    X_test[column] = encoder.transform(X_test[column])



# %%
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# %% [markdown]
# ### Standardize the Features

# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# y_train = scaler.fit_transform(y_train)
# y_test = scaler.transform(y_test)

y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

# %% [markdown]
# ## Regular linear regrission model 

# %%
# Initialize the Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Make predictions
y_pred = linear_model.predict(X_test)

# Calculate Mean Squared Error and R-squared for the test set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)

# Calculate Mean Squared Error for training and testing sets
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("R-squared score:", r2)
print("Train Mean Squared Error:", train_mse)
print("Test Mean Squared Error:", test_mse)

# %% [markdown]
# #### * Looks like we have some problem, the model cannot genaralize well , overfitting problem !!!!

# %% [markdown]
# ## Try lasso model to reduce the overfitting
# 

# %%
# Initialize Lasso Regression model
lasso_model = Lasso(alpha=0.1)  # Adjust the alpha value for regularization

# Train the Lasso model
lasso_model.fit(X_train, y_train)

# Make predictions
y_pred_lasso = lasso_model.predict(X_test)


# Assess the Lasso model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("R-squared score (Lasso):", r2_lasso)

# Calculate Mean Squared Error for training and testing sets with Lasso Regression
train_mse_lasso = mean_squared_error(y_train, lasso_model.predict(X_train))
test_mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print("Train Mean Squared Error (Lasso):", train_mse_lasso)
print("Test Mean Squared Error (Lasso):", test_mse_lasso)

joblib.dump(lasso_model, 'lasso_model_weights.pkl')


# %% [markdown]
# #### * Better results

# %% [markdown]
# # Logistic regrission

# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_f(X, t, weights):
    m = X.shape[0]
    prop = sigmoid(np.dot(X, weights))
    positive = -t * np.log(prop)
    negative = - (1 - t) * np.log(1 - prop)
    cost = 1 / m * np.sum(positive + negative)
    return cost


def f_dervative(X, t, weights):
    m = X.shape[0]
    prop = sigmoid(np.dot(X, weights))
    error = prop - t
    gradient = X.T @ error / m
    return gradient


def gradient_descent(X, t, step_size=0.1, precision=0.0001, max_iter=7000):    
    examples, features = X.shape
    iter = 0
    costs = []  # Move this line outside the loop
    cur_weights = np.random.rand(features)  # random starting point
    last_weights = cur_weights + 100 * precision   

    # print(f'Initial Random Cost: {cost_f(X, t, cur_weights)}')

    while norm(cur_weights - last_weights) > precision and iter < max_iter:
        last_weights = cur_weights.copy()  #  copy
        gradient = f_dervative(X, t, cur_weights)
        cur_weights -= gradient * step_size
        #print(cost_f(X, cur_weights))
        iter += 1
        current_cost = cost_f(X, t, cur_weights)
        costs.append(current_cost)

    # Plot the cost over iterations
    plt.plot(range(1, iter + 1), costs, linestyle='-', color='b')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function vs iterations')
    plt.show()
    print(f'Total Iterations {iter}')
    print(f'Optimal Cost: {cost_f(X, t, cur_weights)}')
    return cur_weights


def accuracy(X, t, weights, threshold = 0.5):
    m = X.shape[0]
    prop = sigmoid(np.dot(X, weights))
    labels = (prop >= threshold).astype(int)
    correct = np.sum((t == labels))
    return correct / m * 100.0

# %%
Xl=data.drop(["Loan_Status","Max_Loan_Amount"],axis=1)
yl=data["Loan_Status"]

# %%
X_trainl, X_testl, y_trainl, y_testl = train_test_split(Xl, yl, test_size=0.2, random_state=42,shuffle=True)

# %%
columns_to_encode = ["Gender", "Married", "Dependents", "Education", "Loan_Tenor", "Credit_History", "Property_Area"]

for column in columns_to_encode:
    encoder = LabelEncoder()
    X_trainl[column] = encoder.fit_transform(X_trainl[column])
    X_testl[column] = encoder.transform(X_testl[column])

encod = LabelEncoder()
y_trainl = encod.fit_transform(y_trainl)

# Transform the 'Loan_Status' column in y_test
y_testl = encod.transform(y_testl)


# %%
numeric_cols = ["Income", "Coapplicant_Income"]
scale = StandardScaler()

# Fit and transform on training set
X_trainl[numeric_cols] = scale.fit_transform(X_trainl[numeric_cols])

# Use the same scaler to transform the test set
X_testl[numeric_cols] = scale.transform(X_testl[numeric_cols])

# %%
weights = gradient_descent(X_trainl,y_trainl)
print(f'Accuracy: {accuracy(X_testl,y_testl,weights)}')
np.save('Logistic_model_weights.npy', weights)

# %% [markdown]
# ## Load the new data and preprocess it

# %%
data_new = pd.read_csv(r"C:\Users\Essam\Desktop\Assignment 1 ML\loan_new.csv")
# Removing spaces in column name
data_new = data_new.dropna()
# Write the cleaned data to a new CSV file
data_new.to_csv(r"C:\Users\Essam\Desktop\Assignment 1 ML\cleaned_data.csv", index=False)

# %%
data_new = data_new.drop(["Loan_ID"],axis=1)

# %%
#encode the columns for new data on the same encoder of train data
for column in columns_to_encode:
    data_new[column] = encod.fit_transform(data_new[column])

# %%
numeric_cols = ["Income", "Coapplicant_Income"]
# Fit on training set using training scale of old data
data_new[numeric_cols] = scale.transform(data_new[numeric_cols])

# %%
loaded_weights = np.load('Logistic_model_weights.npy')
def predict_loan_status(X, weights):
    probabilities = sigmoid(np.dot(X, weights))
    return (probabilities >= 0.5).astype(int)

# Predict loan status for data_new
predicted_loan_status = predict_loan_status(data_new, loaded_weights)
predicted_loan_status

# %%
cleaned_data = pd.read_csv(r"C:\Users\Essam\Desktop\Assignment 1 ML\cleaned_data.csv")

# Adding the predicted values as a new column 'Loan_Status'
cleaned_data['Loan_Status'] = predicted_loan_status
cleaned_data['Loan_Status'] = cleaned_data['Loan_Status'].apply(lambda x: 'Y' if x == 1 else 'N')

# Save the updated data to a new CSV file
cleaned_data.to_csv(r"C:\Users\Essam\Desktop\Assignment 1 ML\cleaned_data.csv", index=False)

# %%
loaded_model = joblib.load('lasso_model_weights.pkl')
predictions_lasso = loaded_model.predict(data_new)
predictions_lasso

# %%
cleaned_data['Max_Loan_Amount'] = np.expm1(predictions_lasso)
cleaned_data.to_csv(r"C:\Users\Essam\Desktop\Assignment 1 ML\cleaned_data.csv", index=False)


