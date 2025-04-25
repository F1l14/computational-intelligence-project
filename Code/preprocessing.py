import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the data
df = pd.read_csv("alzheimers_disease_data.csv")

# Remove first and last column
df = df.iloc[:, 1:-1]


categorical_cols = ['Ethnicity', 'EducationLevel', 'Gender']
numeric_cols_to_scale = ['Age', 'BMI',	'Smoking',	'AlcoholConsumption',	'PhysicalActivity',	'DietQuality',	'SleepQuality', 'SystolicBP',	'DiastolicBP',	'CholesterolTotal',	'CholesterolLDL',	'CholesterolHDL',	'CholesterolTriglycerides',	'MMSE',	'FunctionalAssessment', 'ADL']

# Create the ColumnTransformer
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), categorical_cols),
        ('scaler', StandardScaler(), numeric_cols_to_scale)
    ],
    remainder='passthrough'  # Keeps the rest of the columns unchanged
)

# Fit and transform the data
df_encoded = ct.fit_transform(df)

# Get feature names
encoded_feature_names = ct.named_transformers_['onehot'].get_feature_names_out(categorical_cols)
scaled_feature_names = numeric_cols_to_scale
passthrough_cols = [col for col in df.columns if col not in categorical_cols + numeric_cols_to_scale]

# Combine all column names in correct order
all_columns = list(encoded_feature_names) + scaled_feature_names + passthrough_cols

# Create final DataFrame
df_encoded = pd.DataFrame(df_encoded, columns=all_columns)

# Save
df_encoded.to_excel("preprocessed.xlsx", index=False)
df_encoded.to_csv("preprocessed.csv", index=False)