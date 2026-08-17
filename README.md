# Phân tích cảm xúc bình luận đồ ăn tiếng Việt

Đồ án môn Trí tuệ nhân tạo — Nhóm 16

## Mô tả

Bài toán phân loại cảm xúc (tích cực/tiêu cực) cho các bình luận đánh giá đồ ăn tiếng Việt, gồm tiền xử lý văn bản, trích xuất đặc trưng TF-IDF và huấn luyện, so sánh nhiều mô hình học máy. Có thêm demo web bằng Flask để nhập câu và dự đoán cảm xúc trực tiếp.

## Dữ liệu

- **Nguồn:** [Vietnamese Sentiment Analysis - Food Reviews](https://www.kaggle.com/datasets/guutran/vietnamese-sentiment-analysis-food-reviews) (Kaggle)
- **Nhãn:** Rating — 0 (tiêu cực), 1 (tích cực)

## Quy trình

1. **Tiền xử lý:** làm sạch text (bỏ URL, HTML, ký tự đặc biệt), tách từ tiếng Việt (`pyvi`, `underthesea`), loại bỏ stopwords
2. **Trích xuất đặc trưng:** TF-IDF (5000 features) + độ dài comment đã chuẩn hóa (`StandardScaler`)
3. **Huấn luyện & đánh giá mô hình:**
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Naive Bayes
   - SVM
   - Bonus: tự cài đặt Gaussian Naive Bayes và XGBoost (gradient boosting) từ đầu, không dùng thư viện
4. **Trực quan hóa:** confusion matrix, ROC curve, so sánh F1-score giữa các mô hình
5. **Triển khai demo:** ứng dụng Flask (`app_flask.py`) load model đã lưu (`.pkl`) để dự đoán cảm xúc cho câu nhập vào, chạy trên Colab qua `ngrok`

## Kết quả (F1-score)

| Model | Class 0 (tiêu cực) | Class 1 (tích cực) |
|---|---|---|
| Logistic Regression | 0.78 | 0.92 |
| Random Forest | 0.78 | 0.93 |
| XGBoost | 0.79 | 0.93 |
| Naive Bayes | 0.75 | 0.92 |
| SVM | 0.76 | 0.92 |

XGBoost cho kết quả tốt nhất tổng thể; các mô hình đều dự đoán lớp tích cực tốt hơn lớp tiêu cực do dữ liệu mất cân bằng.

## Công nghệ sử dụng

`pandas`, `numpy`, `scikit-learn`, `underthesea`, `pyvi`, `xgboost`, `matplotlib`/`seaborn`, `Flask`, `joblib`

## Cách chạy

Mở notebook `Project_TTNT_nhóm_16.ipynb` trên Google Colab và chạy tuần tự từ trên xuống (notebook tự tải dataset qua `kagglehub`).
