import streamlit as st
import pandas as pd
import numpy as np
import math
# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
# ML algorithms
from sklearn.linear_model import LinearRegression,LogisticRegression, Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
# Model Selection for Cross Validation
from sklearn.model_selection import train_test_split,StratifiedKFold, KFold,cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
# Machine Learning metrics
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score,mean_squared_error,mean_absolute_error,r2_score
# Hiding warnings
import warnings
warnings.filterwarnings("ignore")
#emojis

st.sidebar.markdown('''
                    # Selection
                    - <a href='#dataset' style='text-decoration:none;font-size:25px;font-weight: 800;'>Dataset</a>
                    <br>
                    - <a href='#prediction-with-models' style='text-decoration:none;font-size:25px;font-weight: 800;'>Prediction with models</a>
                    
                    ''', unsafe_allow_html=True)

st.title("Rain Predict")
# st.set_page_config(page_title=' Raining Predict', page_icon='🌨')

st.balloons()

st.title("🌤 XÂY DỰNG MÔ HÌNH HỒI QUY VÀ DỰ ĐOÁN THỜI TIẾT", )
st.markdown('##')

df= pd.read_csv('C:\@@Learn\Streamlit with python\Data\weatherAUS.csv')

st.header('DATASET')
st.caption('Tập dữ liệu mẫu')
st.dataframe(df)


st.text('''
    Các thuộc tính của Dataset :
    • Date: Ngày tháng năm, Object
    • Location: Vị trí, Object
    • MinTemp: Nhiệt độ tối thiểu, float64
    • MaxTemp: Nhiệt độ tối đa, Object
    • Rainfall: Lượng mưa, float64
    • Evaporation: Bay hơi, float64
    • Sunshine: Ánh sáng mặt trời, float64
    • WindGustDir: Gió, Object
    • WindGustSpeed: Tốc độ gió, Object
    • WindDir9am: Gió mạnh lúc 9am, float64
    • WindDir3pm: Gió mạnh lúc 3pm, Float
    • WindSpeed9am: Tốc độ gió lúc 9am, float64
    • WindSpeed3pm: Tốc độ gió lúc 3pm, float64
    • Humidity9am: Độ ẩm 9 giờ sáng, float64
    • Humidity3pm: Độ ẩm lúc 3 giờ pm, float64
    • Pressure9am: Áp lực lúc 9am, Object
    • Pressure3pm: Áp lực lúc 3pm, Object
    • Cloud9am: Đám mây lúc 9am, float64
    • Cloud3pm: Đám mây lúc 3pm, Object
    • Temp9am: Nhiệt độ lúc 9am, Object
    • Temp3pm: Nhiệt độ lúc 3pm, float64
    • RainToday: Mưa hôm nay, float64
    • RainTomorrow: Mưa ngày mai, float64	
''')

st.header('IMPORT LIBRARY')
st.code('''
#Import library
import datetime
import pandas as pd
import numpy as np
import math
# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
# ML algorithms
from sklearn.linear_model import LinearRegression,LogisticRegression, Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
# Model Selection for Cross Validation
from sklearn.model_selection import train_test_split,StratifiedKFold, KFold,cross_val_score
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
from keras.optimizers import Adam
from sklearn.neighbors import LocalOutlierFactor
# Machine Learning metrics
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score,mean_squared_error,mean_absolute_error,r2_score
from keras import callbacks
# Hiding warnings
import warnings
warnings.filterwarnings("ignore")
#Conect google drive
from google.colab import drive
drive.mount('/content/drive')
        ''')
st.text('Vì thư viện catboost không có sẵn nên sẽ ta sẽ install')
st.code('!pip install catboost')

st.subheader('Thông tin chi tiết về các thuộc tính cũng như kiểu dữ liệu của nó')

st.table(df.dtypes)

st.header('TRỰC QUAN HÓA DỮ LIỆU')

st.subheader('Biểu đồ cột đếm mục tiêu (target)')

def v_line_target():
    cols =['green','blue']
    fig = plt.figure(figsize=(10, 4))
    plot =sns.countplot(x=df['RainTomorrow'], palette= cols)
    st.pyplot(fig)
v_line_target()

st.subheader('Biểu đồ tương quan giữa các thuộc tính')

def v_Hemap():
    a=df.select_dtypes(exclude=['object'])
    corrmat = a.corr()
    cmap= sns.diverging_palette(260, -10, s=50, l=75,n =6, as_cmap=True)
    plt.figure(figsize=(15, 15))
    sns.heatmap(corrmat, cmap= cmap, annot= True, square =True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

v_Hemap()

st.text('Check độ dài của các chuỗi cột "Date"')

lengths= df['Date'].str.len()

value_counts = lengths.value_counts()
st.code(value_counts)

st.text('Chuyển đổi kiểu dữ liệu của cột Date để tách ra cột Year, Month, Day')
# Chuyển type int64 thành datetime của cột Date
df['Date']=pd.to_datetime(df['Date'])
# Tạo 1 cột cho năm (year)
df['year']= df.Date.dt.year

# Mã hóa dữ liệu ngày, tháng theo tham số tuần hoàn để hổ trợ thiết lập mô hình
def encode(data,col, max_val):
  df[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
  df[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
  return df
df['month'] = df.Date.dt.month
df= encode(df,'month',12)

df['day'] = df.Date.dt.day
df= encode(df, 'day',31)

st.code('''
# Chuyển type int64 thành datetime của cột Date
df['Date']=pd.to_datetime(df['Date'])
# Tạo 1 cột cho năm (year)
df['year']= df.Date.dt.year

# Mã hóa dữ liệu ngày, tháng theo tham số tuần hoàn để hổ trợ thiết lập mô hình
def encode(data,col, max_val):
  df[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
  df[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
  return df
df['month'] = df.Date.dt.month
df= encode(df,'month',12)

df['day'] = df.Date.dt.day
df= encode(df, 'day',31)

df.head()
''')

st.dataframe(df.iloc[:, 23:])

st.subheader('Biểu đồ phân phối của ngày trong khoảng 1 năm')
def v_line_d():
    section = df[:360]
    plt.figure(figsize=(10, 6))
    tm = section["day"].plot(color="lightgreen")
    tm.set_title("Distribution Of Days Over Year")
    tm.set_ylabel("Days In month")
    tm.set_xlabel("Days In Year")
    st.pyplot()
v_line_d()

st.subheader('Biểu đồ phân phối chu kỳ của tháng')
def v_scatter_month():
    plt.figure(figsize=(10, 6))
    cyclic_month = sns.scatterplot(x="month_sin",y="month_cos",data=df, color="lightblue")
    cyclic_month.set_title("Cyclic Encoding of Month")
    cyclic_month.set_ylabel("Cosine Encoded Months")
    cyclic_month.set_xlabel("Sine Encoded Months")
    st.pyplot()
v_scatter_month()

st.subheader('Biểu đồ phân phối chu kỳ của ngày')
def v_scatter_day():
    # Biểu đồ phân phối chu kỳ của ngày
    plt.figure(figsize=(10, 6))
    cyclic_day = sns.scatterplot(x='day_sin',y='day_cos',data=df, color="#C2C4E2")
    cyclic_day.set_title("Cyclic Encoding of Day")
    cyclic_day.set_ylabel("Cosine Encoded Day")
    cyclic_day.set_xlabel("Sine Encoded Day")
    st.pyplot()
v_scatter_day()

st.header('TIỀN XỬ LÝ DỮ LIỆU')

st.subheader('Kiểm tra giá trị null của cột object')
st.code('''
for i in object_cols:
    print(i, df[i].isnull().sum())
''')
s = (df.dtypes == "object")
object_cols = list(s[s].index)
for i in object_cols:
    st.write(f"{i}: {df[i].isnull().sum()} missing values")

st.subheader('Điền giá trị trống bằng mode của cột trong khoảng giá trị')
st.code('''
for i in object_cols:
    df[i].fillna(df[i].mode()[0], inplace=True)
''')
for i in object_cols:
    df[i].fillna(df[i].mode()[0], inplace=True)

st.subheader('Kiểm tra giá trị null của cột float64')
st.code('''
for i in num_cols:
    print(i, df[i].isnull().sum())
''')
t = (df.dtypes == "float64")
num_cols = list(t[t].index)
for i in num_cols:
    st.write(f"{i}: {df[i].isnull().sum()} missing values")

st.subheader(' Điền các giá trị thiếu bằng trung bình của cột giá trị')
st.code('''
for i in num_cols:
    df[i].fillna(df[i].median(), inplace=True)
''')
for i in num_cols:
    df[i].fillna(df[i].median(), inplace=True)

st.subheader('Biểu đồ hiển thị lượng mưa hằng năm')
def v_line_rainfall():
    sns.lineplot(x=df['year'], y=df['Rainfall'])
    st.pyplot()
v_line_rainfall()

st.subheader('Biểu đồ thể hiện tốc độ gió của từng năm')
def v_bar_speed():
    plt.figure(figsize=(10,2))
    sns.barplot(x= df['year'], y= df['WindGustSpeed'], palette= sns.color_palette('viridis'))
    st.pyplot()
v_bar_speed()

st.subheader ('Biểu đồ hiển thị nhiệt độ min/max trung bình mõi năm')
def v_line_temp():
    plt.figure(figsize=(12,8))
    average_min_temperature_per_year= df.groupby('year')['MinTemp'].mean()
    average_max_temperature_per_year= df.groupby('year')['MaxTemp'].mean()
    plt.plot(average_min_temperature_per_year.index, average_min_temperature_per_year.values, marker='o', linestyle='-', color='b', label='Min Temp')
    plt.plot(average_max_temperature_per_year.index, average_max_temperature_per_year.values, marker='o', linestyle='-', color='r', label='Max Temp')
    plt.legend()
    st.pyplot()
v_line_temp()

st.subheader('Chuyển dổi dữ liệu object')
st.code('''
label_encoder = LabelEncoder()
for i in object_cols:
    df[i] = label_encoder.fit_transform(df[i])
''')
label_encoder = LabelEncoder()
for i in object_cols:
    df[i] = label_encoder.fit_transform(df[i])

st.subheader('Hiển thị mô hình thống kê các thuộc tính của feature')
features = df.drop(['RainTomorrow', 'Date','month_sin','month_cos','day_sin','day_cos'], axis=1)# dropping target and extra columns

target = df[['RainTomorrow']]

#Set up a standard scaler for the features
col_names = list(features.columns)
s_scaler = StandardScaler()
features = s_scaler.fit_transform(features)
features = pd.DataFrame(features, columns=col_names)

st.dataframe(features.describe().T)

st.subheader('Kiểm tra ngoại lai (outlier)')
def p_v_outlier():
    colours = ["#D0DBEE", "#C2C4E2", "#EED4E5", "#D1E6DC", "#BDE2E2"]
    plt.figure(figsize=(10,5))
    sns.boxenplot(data = features,palette = colours)
    plt.xticks(rotation=90)
    st.pyplot()
p_v_outlier()

st.subheader('Xóa bỏ outlier')
features[["RainTomorrow"]] = target

#Dropping with outlier

features = features[(features["MinTemp"]<2.3)&(features["MinTemp"]>-2.3)]
features = features[(features["MaxTemp"]<2.3)&(features["MaxTemp"]>-2)]
features = features[(features["Rainfall"]<4.5)]
features = features[(features["Evaporation"]<2.8)]
features = features[(features["Sunshine"]<2.1)]
features = features[(features["WindGustSpeed"]<4)&(features["WindGustSpeed"]>-4)]
features = features[(features["WindSpeed9am"]<4)]
features = features[(features["WindSpeed3pm"]<2.5)]
features = features[(features["Humidity9am"]>-3)]
features = features[(features["Humidity3pm"]>-2.2)]
features = features[(features["Pressure9am"]< 2)&(features["Pressure9am"]>-2.7)]
features = features[(features["Pressure3pm"]< 2)&(features["Pressure3pm"]>-2.7)]
features = features[(features["Cloud9am"]<1.8)]
features = features[(features["Cloud3pm"]<2)]
features = features[(features["Temp9am"]<2.3)&(features["Temp9am"]>-2)]
features = features[(features["Temp3pm"]<2.3)&(features["Temp3pm"]>-2)]
def p_v_r_outlier():
    plt.figure(figsize=(10,5))
    sns.boxenplot(data = features)
    plt.xticks(rotation=90)
    st.pyplot()
p_v_r_outlier()

st.header('Xây dựng mô hình')
X = features.drop(['RainTomorrow'], axis = 1)
y=features['RainTomorrow']# Splitting df into X and y # Splitting df into X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,shuffle=True)
#Printing info on X and y
st.text('Info on X')
st.dataframe(X)
st.text('Info on y')
st.dataframe(y)

st.subheader('Linear Regression')
st.text('Tạo mô hình')
st.code('''
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_Linear = model.predict(X_test)
''')

model = LinearRegression()
model.fit(X_train, y_train)
y_pred_Linear = model.predict(X_test)
st.text('Đánh giá mô hình(RMSE, MAE, R-squared)')
mae = mean_absolute_error(y_test, y_pred_Linear)
mse = mean_squared_error(y_test, y_pred_Linear)
r2 = r2_score(y_test, y_pred_Linear)
st.code('''
mae = mean_absolute_error(y_test, y_pred_Linear)
mse = mean_squared_error(y_test, y_pred_Linear)
r2 = r2_score(y_test, y_pred_Linear)
''')

st.write(f"Linear Regression Mean Squared Error: {mse:.2f}")
st.write(f"Linear Regression Mean Absolute Error: {mae:.2f}")
st.write(f"Linear Regression R-squared: {r2:.2f}")

st.subheader('Trực quan hóa mô hình dự đoán')
def v_Linear():
    x_ax=range(len(X_test))
    plt.scatter(x_ax, y_test, s=5, color="blue", label="Điểm thực")
    plt.plot(x_ax, y_pred_Linear, lw=1.5, color="red", label="Điểm dự đoán")
    plt.legend()
    st.pyplot()
v_Linear()

st.subheader('Linear Regression')
st.text('Tạo mô hình')
st.code('''
# Tạo mô hình Lasso
lasso_model = Lasso(alpha=1.0)

# Fit the model to the training data
lasso_model.fit(X_train, y_train)

# Make predictions on the test set
lasso_predictions = lasso_model.predict(X_test)

''')
# Tạo mô hình Lasso
lasso_model = Lasso(alpha=1.0)

# Fit the model to the training data
lasso_model.fit(X_train, y_train)

# Make predictions on the test set
lasso_predictions = lasso_model.predict(X_test)
st.text('Đánh giá mô hình(RMSE, MAE, R-squared)')
mse = mean_squared_error(y_test, lasso_predictions)
mae = mean_absolute_error(y_test, lasso_predictions)
r2 = r2_score(y_test, lasso_predictions)
st.code('''
mse = mean_squared_error(y_test, lasso_predictions)
mae = mean_absolute_error(y_test, lasso_predictions)
r2 = r2_score(y_test, lasso_predictions)
''')

st.write(f"Lasso Regression Mean Squared Error: {mse:.2f}")
st.write(f"Lasso Regression Mean Absolute Error: {mae:.2f}")
st.write(f"Lasso Regression R-squared: {r2:.2f}")

# Define the parameter grid
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Initialize Lasso Regression model
lasso_model = Lasso()

# Initialize GridSearchCV
lasso_grid_search = GridSearchCV(lasso_model, param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_lasso_params = lasso_grid_search.best_params_

# Initialize Lasso model with best hyperparameters
best_lasso_model = Lasso(alpha=best_lasso_params['alpha'])
# Fit the Lasso model with the best hyperparameters
best_lasso_model.fit(X_train, y_train)

# Perform k-fold cross-validation
lasso_cv_scores = cross_val_score(best_lasso_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
lasso_rmse_scores = np.sqrt(-lasso_cv_scores)

lasso_predictions_k= best_lasso_model.predict(X_test)
st.text('Fit Lasso model với the best hyperparameters')
st.code('''
# Define the parameter grid
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Initialize Lasso Regression model
lasso_model = Lasso()

# Initialize GridSearchCV
lasso_grid_search = GridSearchCV(lasso_model, param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_lasso_params = lasso_grid_search.best_params_

# Initialize Lasso model with best hyperparameters
best_lasso_model = Lasso(alpha=best_lasso_params['alpha'])
# Fit the Lasso model with the best hyperparameters
best_lasso_model.fit(X_train, y_train)

# Perform k-fold cross-validation
lasso_cv_scores = cross_val_score(best_lasso_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
lasso_rmse_scores = np.sqrt(-lasso_cv_scores)
''')
st.text('Đánh giá lại mô hình')

# Print the metrics
lasso_predictions_k= best_lasso_model.predict(X_test)
st.write(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, lasso_predictions_k)):.3f}")
st.write(f"Test MAE: {mean_absolute_error(y_test, lasso_predictions_k):.3f}")
st.write(f"Test R-squared: {r2_score(y_test, lasso_predictions_k):.3f}")

st.subheader('Trực quan mô hình LASSO')
def v_lasso():
    x_ax=range(len(X_test))
    plt.plot(x_ax, lasso_predictions_k, lw=1.5, color="red", label="Điểm dự đoán", zorder=1)
    plt.scatter(x_ax, y_test, s=5, color="blue", label="Điểm thực", zorder=2)
    plt.legend()
    st.pyplot()
v_lasso()

st.header('Ridge Regression')
st.text('Tạo mô hình Ridge')
st.code('''
#Tạo mô hình ridge
ridge_model = Ridge(alpha=1.0)

# Fit the model to the training data
ridge_model.fit(X_train, y_train)

# Make predictions on the test set
ridge_predictions = ridge_model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, ridge_predictions)
mae = mean_absolute_error(y_test, ridge_predictions)
r2 = r2_score(y_test, ridge_predictions)
''')
#Tạo mô hình ridge
ridge_model = Ridge(alpha=1.0)

# Fit the model to the training data
ridge_model.fit(X_train, y_train)

# Make predictions on the test set
ridge_predictions = ridge_model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, ridge_predictions)
mae = mean_absolute_error(y_test, ridge_predictions)
r2 = r2_score(y_test, ridge_predictions)

st.text('Đánh giá mô hình Ridge')
st.write(f"Ridge Regression Mean Squared Error: {mse:.2f}")
st.write(f"Ridge Regression Mean Absolute Error: {mae:.2f}")
st.write(f"Ridge Regression R-squared: {r2:.2f}")

st.text('Fit model Ridge với best hyperparameters')
st.code('''
# Define the parameter grid
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Initialize Ridge Regression model
ridge_model = Ridge()

# Initialize GridSearchCV
ridge_grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_ridge_params = ridge_grid_search.best_params_

# Initialize Ridge model with best hyperparameters
best_ridge_model = Ridge(alpha=best_ridge_params['alpha'])
# Fit the Ridge model with the best hyperparameters
best_ridge_model.fit(X_train, y_train)

# Perform k-fold cross-validation
ridge_cv_scores = cross_val_score(best_ridge_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
ridge_rmse_scores = np.sqrt(-ridge_cv_scores)
        ''')

# Define the parameter grid
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Initialize Ridge Regression model
ridge_model = Ridge()

# Initialize GridSearchCV
ridge_grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_ridge_params = ridge_grid_search.best_params_

# Initialize Ridge model with best hyperparameters
best_ridge_model = Ridge(alpha=best_ridge_params['alpha'])
# Fit the Ridge model with the best hyperparameters
best_ridge_model.fit(X_train, y_train)

# Perform k-fold cross-validation
ridge_cv_scores = cross_val_score(best_ridge_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
ridge_rmse_scores = np.sqrt(-ridge_cv_scores)

st.text('Đánh giá lại mô hình Ridge')
ridge_predictions_k = best_ridge_model.predict(X_test)
st.write(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, ridge_predictions_k)):.3f}")
st.write(f"Test MAE: {mean_absolute_error(y_test, ridge_predictions_k):.3f}")
st.write(f"Test R-squared:, {r2_score(y_test, ridge_predictions_k):.3f}")

st.text('Trực quan hóa mô hình')
def v_ridge():
    x_ax=range(len(X_test))
    plt.scatter(x_ax, y_test, s=5, color="blue", label="Điểm thực",zorder =2)
    plt.plot(x_ax, ridge_predictions_k, lw=1.5, color="green", label="Điểm dự đoán", zorder =1)
    plt.legend()
    st.pyplot()
v_ridge()

# st.header("Logistic Regression")
# st.text('Tạo mô hình')
# st.code('''
# # Tạo mô hình hồi quy
# model_Logistic = LogisticRegression(solver='liblinear')
# model_Logistic.fit(X_train,y_train)
# #Dự đoán
# y_predict_Logistic = model_Logistic.predict(X_test)
# r2 = r2_score(y_predict_Logistic,y_test)
# x_ax=range(len(X_test))
# mean_adjusted_r2 = np.mean(r2)
# mean = mean_squared_error(y_predict_Logistic,y_test)
# trainScore = math.sqrt(mean)

# ''')
# # Tạo mô hình hồi quy
# model_Logistic = LogisticRegression(solver='liblinear')
# model_Logistic.fit(X_train,y_train)
# #Dự đoán
# y_predict_Logistic = model_Logistic.predict(X_test)
# r2 = r2_score(y_predict_Logistic,y_test)
# x_ax=range(len(X_test))
# mean_adjusted_r2 = np.mean(r2)
# mean = mean_squared_error(y_predict_Logistic,y_test)
# trainScore = math.sqrt(mean)

# st.text('Đánh giá mô hình')
# st.write(f'RMSE :,{trainScore:.2f}')
# st.write(f'Mean Adjusted R-squared = {mean_adjusted_r2:.2f}')
# st.write(f"Xác định độ chính xác,{accuracy_score(y_test,y_predict_Logistic):.2f}")

# st.text('Fit model Logistic với best hyperparameters')
# st.code('''
# # Define the parameter grid
# param_grid = {'n_jobs': [0.001, 0.01, 0.1, 1, 10, 100]}

# # Initialize Ridge Regression model
# logis_model = LogisticRegression()

# # Initialize GridSearchCV
# logis_grid_search = GridSearchCV(logis_model, param_grid, cv=5, scoring='neg_mean_squared_error')
# logis_grid_search.fit(X_train, y_train)

# # Get the best hyperparameters
# best_logis_params = logis_grid_search.best_params_

# # Initialize Ridge model with best hyperparameters
# best_logis_model = LogisticRegression(n_jobs=best_logis_params['n_jobs'])
# # Fit the Ridge model with the best hyperparameters
# best_logis_model.fit(X_train, y_train)

# # Perform k-fold cross-validation
# logis_cv_scores = cross_val_score(best_logis_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
# logis_rmse_scores = np.sqrt(-logis_cv_scores)
# logis_predictions_k = best_logis_model.predict(X_test)
# ''')
# # Define the parameter grid
# param_grid = {'n_jobs': [0.001, 0.01, 0.1, 1, 10, 100]}

# # Initialize Ridge Regression model
# logis_model = LogisticRegression()

# # Initialize GridSearchCV
# logis_grid_search = GridSearchCV(logis_model, param_grid, cv=5, scoring='neg_mean_squared_error')
# logis_grid_search.fit(X_train, y_train)

# # Get the best hyperparameters
# best_logis_params = logis_grid_search.best_params_

# # Initialize Ridge model with best hyperparameters
# best_logis_model = LogisticRegression(n_jobs=best_logis_params['n_jobs'])
# # Fit the Ridge model with the best hyperparameters
# best_logis_model.fit(X_train, y_train)

# Perform k-fold cross-validation
# logis_cv_scores = cross_val_score(best_logis_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
# logis_rmse_scores = np.sqrt(-logis_cv_scores)
# logis_predictions_k = best_logis_model.predict(X_test)

# st.text('Đánh giá lại mô hình')
# st.write(f"Test RMSE:, {np.sqrt(mean_squared_error(y_test, logis_predictions_k)):.2f}")
# st.write(f"Test MAE:, {mean_absolute_error(y_test, logis_predictions_k):.2f}")
# st.write(f"Test R-squared:, {r2_score(y_test, logis_predictions_k):.2f}")

# st.text('Trực quan hóa mô hình')
# def v_logic():
#     plt.scatter(x_ax, y_test, s=5, color="red", label="Điểm thực",zorder= 2)
#     plt.plot(x_ax, logis_predictions_k, lw=1.5, color="green", label="Điểm dự đoán",zorder =1)
#     plt.legend()
#     st.pyplot()
# v_logic()

st.header('PREDICTION WITH MODELS')
st.subheader('Điền dữu liệu để dự đoán')
require_data_pre=[]

New_value=[2,	13.4,	22.9,	0.6,	4.8,	8.4,	13,	44.0,	13,	14,	20.0,	24.0,	71.0,	22.0,	1007.7,	1007.1,	8.0,	5.0,	16.9,	21.8, 0, 2008,	12,	1]

for i,z in zip(X_train.columns, New_value):
    user_input = st.number_input(i)
    require_data_pre.append(st.number_input(i,value=z))
st.write(require_data_pre)

require_data_pre = np.array(require_data_pre).reshape(1,-1)
require_data_pre= s_scaler.transform(require_data_pre)

st.subheader("Kết quả dự đoán của các mô hình")
st.caption('Linear Regression')
require_data_pre_linear = model.predict(require_data_pre)

threshold = 0.5
binary_ex_predict = (require_data_pre_linear >= threshold).astype(int)
st.write(f"Predictions (continuous):, {require_data_pre_linear[0]:.2f}")
st.write(f"Predictions (binary):, {binary_ex_predict[0]:.2f}")

st.caption('LASSO Regression')
LA_predict = best_lasso_model.predict(require_data_pre)
st.write(f'Prediction: ,{LA_predict[0]:.2f}')

st.caption('Ridge Regression')
Rd_predict = best_ridge_model.predict(require_data_pre)
st.write(f'Prediction: ,{Rd_predict[0]:.2f}')
