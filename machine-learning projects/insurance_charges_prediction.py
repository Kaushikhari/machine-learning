import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# dropna, replace, astype, strip, str
# Cleaning our dataset
insurance_data_path = 'insurance.csv'
insurance = pd.read_csv(insurance_data_path)
# creating a copy properly
insurance_cleaned = insurance.dropna().copy()
# making sure if there are only 4 regions
insurance_cleaned['region'] = insurance_cleaned['region'].str.lower()
# to standardize all the genders
FEMALE = 'female'
MALE = 'male'
gender_map = {'F': FEMALE, 'woman': FEMALE, 'man': MALE, 'M': MALE}
insurance_cleaned['sex'] = insurance_cleaned['sex'].replace(gender_map)
# to make the smoker column bool
insurance_cleaned['smoker'] = insurance_cleaned['smoker'] == 'yes'
# taking of $
insurance_cleaned['charges'] = insurance_cleaned['charges'].str.strip('$').astype('float64')
# using abs to remove negative values
insurance_cleaned['age'] = insurance_cleaned['age'].abs()

# fitting the model
# preparing the dataset
new_df = pd.get_dummies(insurance_cleaned, prefix=['region'], columns=['region'])
new_df = new_df.drop(columns=['region_southeast'])
new_df['smoker'] = new_df['smoker'].astype('int64')

new_df['is_male'] = (new_df['sex'] == 'male').astype('int64')
new_df = new_df.drop(columns=['sex'])
new_df = new_df.dropna().reset_index(drop=True)
# print(new_df.info())

# Split features and target
X = new_df.drop(columns=['charges'])
y = new_df['charges']

# Train-test split method
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Step 6: Predict on new input (example)
# Create a sample input row (with same column structure as X)
sample_input = X.iloc[0:1]
predicted_charge = model.predict(sample_input)
print("Predicted charges for sample input:", predicted_charge[0])

val_df = pd.read_csv('validation_dataset.csv')


def preprocess(df):
    df_new = pd.get_dummies(df, prefix=['region'], columns=['region'])
    df_new = df_new.drop(columns=['region_southeast'])
    df_new['smoker'] = df_new['smoker'] == 'yes'
    df_new['smoker'] = df_new['smoker'].astype('int64')
    df_new['is_male'] = (df_new['sex'] == 'male').astype('int64')
    df_new = df_new.drop(columns=['sex'])
    return df_new


input_val = preprocess(val_df)
predictions = model.predict(input_val)
validation_data = val_df.copy()

validation_data['predicted_charges'] = predictions

validation_data.loc[validation_data['predicted_charges'] < 1000, 'predicted_charges'] = 1000

print(validation_data.head())



