import pandas as pd

ds = pd.read_csv('diabetes.csv')

diabetes_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction","Age"]
x = ds[diabetes_features]
diabetes_output = ["Outcome"]
y = ds[diabetes_output]

# split the data into 4 categories
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size=0.2)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputed_x_train = pd.DataFrame(imputer.fit_transform(x_train))
imputed_x_test = pd.DataFrame(imputer.transform(x_test))
# Imputation removed column names; put them back
imputed_x_train.columns = x_train.columns
imputed_x_test.columns = x_test.columns

# fit the decision tree model with the data
#   then find the the mean absolute error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
diabetes_model = DecisionTreeRegressor()
diabetes_model.fit(imputed_x_train, y_train)
x_validation = diabetes_model.predict(imputed_x_test)
decision_tree_error = mean_absolute_error( y_test ,x_validation)
print(decision_tree_error)

# Random forest model
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=0)
forest_model.fit(imputed_x_train, y_train.values.ravel())
x_validation_forest = forest_model.predict(imputed_x_test)
random_forest_error = mean_absolute_error(y_test, x_validation_forest)
print(random_forest_error)

# Logistic regression model
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter= 180)
logistic_model.fit(imputed_x_train, y_train.values.ravel())
x_validation_logistic = logistic_model.predict(imputed_x_test)
logistic_error = mean_absolute_error(y_test, x_validation_logistic)
print(logistic_error)

# KNN model
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
#feature scaling 
x_scale = StandardScaler()
imputed_x_train = x_scale.fit_transform(imputed_x_train)
imputed_x_test = x_scale.transform(imputed_x_test)
KNN_model = KNeighborsClassifier(n_neighbors= 27, p=2, metric='euclidean')
KNN_model.fit(imputed_x_train, y_train.values.ravel())
x_validation_KNN = KNN_model.predict(imputed_x_test)
cm = confusion_matrix(y_test, x_validation_KNN)
print(cm)


print(f"the f1 score for KNN is {f1_score(y_test, x_validation_KNN)}")
print(f"the accuracy for KNN is {accuracy_score(y_test, x_validation_KNN)} ")
print(f"the f1 score for decision tree is {f1_score(y_test, x_validation)}")
print(f"the accuracy for decision tree is {accuracy_score(y_test, x_validation)} ")
print(f"the f1 score for logistic is {f1_score(y_test, x_validation_logistic)}")
print(f"the accuracy for logistic is {accuracy_score(y_test, x_validation_logistic)} ")