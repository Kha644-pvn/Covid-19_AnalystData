# 1. Import thư viện
### Import Dependencies
# — Data & Math ————————————————————————————————————
import pandas as pd
import numpy as np

# — Visualization ——————————————————————————————————
import matplotlib.pyplot as plt
import seaborn as sns

# — ML & Preprocessing —————————————————————————————
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.inspection import permutation_importance

# — Clustering —————————————————————————————————————
from dtaidistance import dtw as dtw_lib
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

# — Utilities ——————————————————————————————————————
import joblib
import json, os, warnings
warnings.filterwarnings('ignore')
# 2. Data Pipeline
# 3. Phân tích Clusting Code __ DTW + BIRCH
# 4. Phân tích thuật toán ARIMA
# 5. Kết quả
