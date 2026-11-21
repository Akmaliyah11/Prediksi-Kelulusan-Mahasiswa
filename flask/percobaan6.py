from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Kamu bisa ganti parameternya sesuai eksperimenmu
best_knn = KNeighborsClassifier(n_neighbors=6)
best_svm = SVC(C=5, kernel='rbf', probability=True, class_weight='balanced')
best_rf = RandomForestClassifier(n_estimators=100, random_state=42)

from percobaan6 import best_knn, best_svm, best_rf