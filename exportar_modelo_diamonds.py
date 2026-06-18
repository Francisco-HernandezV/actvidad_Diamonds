import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

URL_DATASET = 'https://raw.githubusercontent.com/Francisco-HernandezV/diamonds/refs/heads/main/diamonds.csv'

data = pd.read_csv(URL_DATASET)

encoders = {}
for col in ['cut', 'color', 'clarity']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

X = data.drop(columns=['Unnamed: 0', 'price'])
y = data['price']

selector_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
selector = SelectFromModel(selector_model, threshold='median')
selector.fit(X, y)
selected_features = list(X.columns[selector.get_support()])
X = X[selected_features]
print("Características seleccionadas:", selected_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=2, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R²  : {r2:.4f}")
print(f"MAE : {mae:.2f}")

encoders_finales = {col: enc for col, enc in encoders.items() if col in selected_features}

bundle = {
    'model': model,
    'encoders': encoders_finales,
    'feature_columns': selected_features,
    'metrics': {'r2': float(r2), 'mae': float(mae)}
}
joblib.dump(bundle, 'diamonds_price_bundle.pkl')
print("Modelo exportado: diamonds_price_bundle.pkl")