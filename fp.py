# Import library yang diperlukan
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Membaca data
datatrain = pd.read_csv("Data train balance.csv")
datatest = pd.read_csv("Data test balance.csv")

# Mengonversi kolom TX_FRAUD menjadi faktor (kategori)
datatrain['TX_FRAUD'] = datatrain['TX_FRAUD'].astype('category')
datatest['TX_FRAUD'] = datatest['TX_FRAUD'].astype('category')

# Mengencode label
le = LabelEncoder()
datatrain['TX_FRAUD'] = le.fit_transform(datatrain['TX_FRAUD'])
datatest['TX_FRAUD'] = le.transform(datatest['TX_FRAUD'])

# Membuat dan melatih model pohon keputusan untuk klasifikasi
model_classification = DecisionTreeClassifier()
model_classification.fit(datatrain[['TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']], datatrain['TX_FRAUD'])

# Menampilkan ringkasan model
print(model_classification)

# Memprediksi menggunakan model
predictions_classification = model_classification.predict(datatest[['TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']])

# Evaluasi performa model
confusion_matrix_classification = confusion_matrix(datatest['TX_FRAUD'], predictions_classification)
print(confusion_matrix_classification)

accuracy_classification = accuracy_score(datatest['TX_FRAUD'], predictions_classification)
print(f"Accuracy: {accuracy_classification}")

# Menghitung ROC dan AUC
fpr, tpr, _ = roc_curve(datatest['TX_FRAUD'], predictions_classification)
roc_auc = auc(fpr, tpr)

# Menggambar ROC
plt.figure()
plt.plot(fpr, tpr, color='red', linestyle='--')
plt.plot([0, 1], [0, 1], color='blue', linestyle='-')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.show()

print(f"Luas AUC: {roc_auc}")

# Evaluasi matriks training
trainpred = model_classification.predict(datatrain[['TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']])
confusion_matrix_train = confusion_matrix(datatrain['TX_FRAUD'], trainpred)
print(confusion_matrix_train)

# Membuat dan melatih model Random Forest
model_rf = RandomForestClassifier()
model_rf.fit(datatrain[['TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']], datatrain['TX_FRAUD'])

# Menampilkan hasil model
print(model_rf)

# Melakukan prediksi menggunakan Random Forest
predict_result_rf = model_rf.predict(datatest[['TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']])

# Menampilkan hasil prediksi
comparation_result_rf = pd.DataFrame({'prediction': predict_result_rf, 'actual': datatest['TX_FRAUD']})
print(comparation_result_rf)

# Menghitung kinerja
confusion_matrix_rf = confusion_matrix(datatest['TX_FRAUD'], predict_result_rf)
print(confusion_matrix_rf)

# Menghitung ROC dan AUC untuk Random Forest
fpr_rf, tpr_rf, _rf = roc_curve(datatest['TX_FRAUD'], predict_result_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Menggambar ROC untuk Random Forest
plt.figure()
plt.plot(fpr_rf, tpr_rf, color='red', linestyle='--')
plt.plot([0, 1], [0, 1], color='blue', linestyle='-')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Random Forest')
plt.show()

print(f"Luas AUC: {roc_auc_rf}")

# Evaluasi matriks training untuk Random Forest
trainpred_rf = model_rf.predict(datatrain[['TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']])
confusion_matrix_trainrf = confusion_matrix(datatrain['TX_FRAUD'], trainpred_rf)
print(confusion_matrix_trainrf)
