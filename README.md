# FetalStateClassification-ML-Project
Cardiotocography (CTG) is a standard prenatal monitoring technique that records fetal heart rate (FHR) and uterine contractions (UC) to assess fetal well-being during pregnancy. Detecting abnormal fetal states early is crucial for reducing risks of complications during labor and delivery. The CTG dataset contains 21 quantitative features extracted from FHR and UC signals collected from 2,126 instances. Each instance is labeled by medical experts with the fetal state (NSP): Normal, Suspect, or athologic. This makes the dataset ideal for supervised multi-class classification tasks, where the goal is to predict fetal health based on CTG signal features.

# Data Information
The data is obtained from Campos, D. & Bernardes, J. (2000). Cardiotocography [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51S4N.

## Features
2126 fetal cardiotocograms (CTGs) were automatically processed and the respective diagnostic features measured. The CTGs were also classified by three expert obstetricians and a consensus classification label assigned to each of them. Classification was both with respect to a morphologic pattern (A, B, C. ...) and to a fetal state (N, S, P). Therefore the dataset can be used either for 10-class or 3-class experiments.

**Feature Names are as follows:**
LB - FHR baseline (beats per minute)
AC - # of accelerations per second
FM - # of fetal movements per second
UC - # of uterine contractions per second
DL - # of light decelerations per second
DS - # of severe decelerations per second
DP - # of prolongued decelerations per second
ASTV - percentage of time with abnormal short term variability
MSTV - mean value of short term variability
ALTV - percentage of time with abnormal long term variability
MLTV - mean value of long term variability
Width - width of FHR histogram
Min - minimum of FHR histogram
Max - Maximum of FHR histogram
Nmax - # of histogram peaks
Nzeros - # of histogram zeros
Mode - histogram mode
Mean - histogram mean
Median - histogram median
Variance - histogram variance
Tendency - histogram tendency
CLASS - FHR pattern class code (1 to 10) 

**Target Feature**
NSP - fetal state class code (N=normal; S=suspect; P=pathologic)