# 🚢 Titanic Survival Prediction: Machine Learning from Scratch vs. Frameworks

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📌 Project Overview
โปรเจกต์ศึกษาและพัฒนาอัลกอริทึม **Machine Learning และ Deep Learning** เพื่อทำนายการรอดชีวิตของผู้โดยสารบนเรือไททานิค (จากชุดข้อมูล OpenML Titanic จำนวน 1,309 รายการ) 

ความโดดเด่นของโปรเจกต์นี้คือการ **เขียนอัลกอริทึมพื้นฐานขึ้นมาเองตั้งแต่ต้นด้วย NumPy (Implementation from Scratch)** เพื่อแสดงการทำงานของสมการคณิตศาสตร์เบื้องหลัง (เช่น Gradient Descent, Backpropagation, และ Gini Impurity) ก่อนนำไปเปรียบเทียบประสิทธิภาพกับไลบรารีมาตรฐานอย่าง **TensorFlow/Keras** และ **Scikit-Learn**

---

## 🛠️ Data Preprocessing & Feature Engineering
1. **Data Cleaning & Imputation:**
   * ตัดคอลัมน์ที่ไม่เกี่ยวข้องกับการจำแนกคลาส (`name`, `ticket`, `home.dest`) และคอลัมน์ที่มีค่าว่าง (Missing Values) จำนวนมาก (`cabin`, `body`, `boat`) ออก
   * **Grouped Median Imputation:** เติมค่าว่างของอายุ (`age`) ด้วยค่ามัธยฐาน (Median) ที่จัดกลุ่มตามระดับชั้นตั๋วและเพศ (`groupby(['pclass', 'sex'])['age'].transform('median')`) เพื่อความแม่นยำทางประชากรศาสตร์
   * เติมค่าว่างของราคาตั๋ว (`fare`) ด้วยค่า Median และท่าเรือที่ขึ้น (`embarked`) ด้วยค่าฐานนิยม (Mode = `'S'`)
2. **Feature Creation & Encoding:**
   * สร้างตัวแปรใหม่ **`is_alone`** จากการรวม `sibsp` และ `parch` เพื่อระบุว่าผู้โดยสารเดินทางคนเดียวหรือไม่
   * แปลงข้อมูลหมวดหมู่ด้วย Label Mapping (`sex`) และ One-Hot Encoding (`embarked_Q`, `embarked_S`) ได้ตัวแปรนำเข้า (Features) ทั้งหมด **7 ตัวแปร**: `['pclass', 'sex', 'age', 'fare', 'embarked_Q', 'embarked_S', 'is_alone']`
3. **Custom Standardization & Train-Test Split:**
   * แบ่งข้อมูลแบบสุ่มในอัตราส่วน **80% Train / 20% Test** พร้อมทำ Z-score Standardization

---

## 🧠 Models Implemented

### 1. From-Scratch Implementations (Pure NumPy)
* **Linear & Logistic Regression:** พัฒนาฟังก์ชัน Sigmoid, การคำนวณ Binary Cross-Entropy Loss และการปรับค่าพารามิเตอร์ ($w, b$) ด้วย Gradient Descent (1,000 Epochs, $\alpha = 0.01$)
* **Multi-Layer Perceptron (MLP from Scratch):** สร้างโครงข่ายประสาทเทียมที่มี Hidden Layer ขนาด 16 โหนด พร้อมเขียนกระบวนการ Forward Pass และ Backward Propagation ด้วยตนเอง
* **CART Decision Tree Classifier (From Scratch):** 
  * เขียนฟังก์ชันคำนวณค่า **Gini Impurity** และอัลกอริทึมค้นหาจุดตัดที่ดีที่สุด (`best_split`)
  * สร้างโครงสร้างต้นไม้แบบ Recursive (`build_tree`, `max_depth=3`) พร้อมเขียนฟังก์ชันวาดแผนผังต้นไม้ด้วย Matplotlib (`plot_custom_tree`)

### 2. Framework Implementations
* **Deep Binary MLP (TensorFlow / Keras):** โครงข่ายประสาทเทียม 4 ชั้น (`Dense 32 -> 16 -> 8 -> 1`) ผสานเทคนิคป้องกัน Overfitting ด้วย **L2 Regularization (`0.001`)**, **BatchNormalization**, และ **Dropout (`0.2`)** ขับเคลื่อนด้วย Adam Optimizer ($\alpha = 0.001$, 150 Epochs)
* **Scikit-Learn Decision Tree:** ใช้ `DecisionTreeClassifier` (`criterion='gini'`) เพื่อตรวจสอบและเปรียบเทียบโครงสร้างการแตกกิ่งกับต้นไม้ที่เขียนขึ้นเอง

---

## 📊 Experimental Results & Model Comparison

| Model / Algorithm | Implementation | Key Hyperparameters | Test Accuracy (%) |
| :--- | :--- | :--- | :---: |
| **Logistic Regression** | NumPy (From Scratch) | `epochs=1000`, `lr=0.01` | **78.24%** |
| **Multi-Layer Perceptron (1-Hidden Layer)** | NumPy (From Scratch) | `hidden_units=16`, `epochs=1000`, `lr=0.01` | **74.43%** |
| **Regularized Deep MLP** | TensorFlow / Keras | `layers=[32,16,8,1]`, `L2`, `BatchNorm`, `Dropout(0.2)` | **78.63%** |
| **Custom Decision Tree (CART)** | **NumPy (From Scratch)** | **`criterion='gini'`, `max_depth=3`** | **82.82%** 🏆 |

> 💡 **Key Takeaway:** โมเดล **Custom Decision Tree ที่เขียนขึ้นเองด้วย NumPy** ให้ความแม่นยำบนชุดข้อมูลทดสอบสูงที่สุดถึง **82.82%** เนื่องจากโครงสร้างแบบ Tree-based สามารถจับความสัมพันธ์แบบไม่เป็นเชิงเส้น (Non-linear interactions) ระหว่างเพศ (`sex`), ชั้นตั๋ว (`pclass`), และช่วงอายุ (`age`) ในข้อมูลรูปแบบตาราง (Tabular Data) ได้อย่างมีประสิทธิภาพ

---

## 🖼️ Visualizations

### Custom Decision Tree Visualization (Built from Scratch)
![Custom Decision Tree](assets/custom_decision_tree.png)

### Deep MLP Training & Validation Loss (TensorFlow/Keras)
![Keras Loss Curve](assets/keras_loss_curve.png)

---

## 💻 How to Run
1. Clone this repository:
   ```bash
   git clone [https://github.com/yourusername/titanic-survival-ml-from-scratch.git](https://github.com/yourusername/titanic-survival-ml-from-scratch.git)
   ```
2. Install required dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib tensorflow
   ```
3. Open and run `titanic_ml_from_scratch.ipynb` in Jupyter Notebook or Google Colab.
