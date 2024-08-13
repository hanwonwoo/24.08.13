import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

header = ['preg', 'plas', 'pres', 'skin','test', 'mass', 'pedi', 'age', 'class'
          ]

data = pd.read_csv("./data/pima-indians-diabetes.data.csv",
                    names= header)

#데이터 전처리 : Min-Max 스케일링
array = data.values
X = array[:, 0 : 8]
Y = array[:, 8]
scaler = MinMaxScaler(feature_range=(0,1))
rescaled_X = scaler.fit_transform(X)

# #데이터 분할
# X_train, X_test, Y_train, Y_test = train_test_split(rescaled_X, Y,test_size=0.2)

#모델 선택 및 분할
model =LogisticRegression()

fold = KFold(n_splits=10, shuffle=True)
acc = cross_val_score(model, rescaled_X, Y, cv=fold, scoring='accuracy')


s= sum(acc)
l = len(acc)
avg = s / l
print(sum(acc))







#모델 선택 및 학습
# model = DecisionTreeClassifier(max_depth=1000, min_samples_split=50, min_samples_leaf=5)
# model.fit(X_train, Y_train)

# #예측값 생성
# y_pred = model.predict(X_test)
#
#
# #모델 정확도
# acc = accuracy_score(Y_test, y_pred)
# print(acc)
# #예측 정확도 확인
# acc = accuracy_score(y_pred_binary, Y_test)
# print(acc)
#
# #Y test 값 저장 및 Y 예측값 저장
# df_Y_test = pd.DataFrame(Y_test)
# df_Y_pred_binary = pd.DataFrame(y_pred_binary)
# # df_Y_test.to_csv("./results/y_test.csv")
# # df_Y_pred_binary.to_csv("./results/y_pred.csv")
#
#
# # print(X_train.shape, X_test.shape, Y_test.shape, Y_test.shape)
#
# #결과(모델 예측값 vs 실체값)시각화
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Actual Values', marker='o')
# plt.scatter(range(len(y_pred_binary)), y_pred_binary, color='red', label='predicted Values', marker='x')
#
# plt.title("comparison of Actual and Predicted values")
# plt.xlabel("Index")
# plt.ylabel("Class (0 or 1)")
# plt.legend()
# plt.show()
# plt.savefig('./result/scatter.png')
#
#
# #예측성 생성
# y_pred = model.predict(X_test)
#
# #모델정확도 계산
# acc = accuracy_score(Y_test, y_pred)
# print(acc)