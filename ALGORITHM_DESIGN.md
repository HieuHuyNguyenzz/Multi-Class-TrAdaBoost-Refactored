# Tài liệu Chi tiết Thiết kế Thuật toán: Gated TrAdaBoost

Tài liệu này cung cấp cái nhìn chi tiết toàn diện về toàn bộ logic triển khai trong codebase, bao gồm các biến, cấu trúc dữ liệu và quy trình tính toán.

---

## 1. Quản lý Cấu hình và Siêu tham số (`src/config.py`)

Hệ thống sử dụng một file config tập trung để quản lý toàn bộ các hằng số.

### 1.1 Thông số Dữ liệu & Model
| Biến | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| `NUM_FEATURE` | 128 | Số lượng đặc trưng của mỗi gói tin. |
| `PACKET_NUM` | 20 | Số lượng gói tin trong một flow. |
| `NUM_CLASSES` | 3 | Số lượng lớp phân loại traffic. |
| `NUM_ESTIMATORS`| 10 | Số lượng weak learners (experts) trong ensemble. |
| `BATCH_SIZE` | 32 | Kích thước batch cho huấn luyện. |
| `NUM_EPOCHS` | 50 | Số epoch huấn luyện cho mỗi weak learner. |
| `CLIENT_LR` | 1e-4 | Tốc độ học cho các weak learners. |

### 1.2 Thông số Gating Network
| Biến | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| `GATING_K` | 5 | Số lượng experts được chọn trong Sparse Inference. |
| `GATING_TAU` | 1.0 | Nhiệt độ cho hàm softmax (Temperature). |
| `GATING_LR` | 1e-4 | Tốc độ học của Gating Network. |
| `GATING_EPOCHS` | 30 | Số epoch huấn luyện Gating Network. |
| `GATING_WEIGHT_DECAY` | 1e-2 | Hệ số regularization để tránh overfitting. |
| `GATING_GRAD_CLIP` | 1.0 | Ngưỡng cắt gradient để ổn định huấn luyện. |
| `GATING_LAMBDA_LB` | 0.01 | Trọng số của Load Balancing Loss. |

### 1.3 Chiến lược Chia dữ liệu miền Đích
| Biến | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| `TARGET_TEST_RATIO` | 0.90 | 90% dữ liệu target dùng làm tập Test. |
| `TARGET_TRAIN_LABELED_RATIO` | 0.1 | 10% của 10% dữ liệu còn lại sẽ có nhãn (Labeled). |

---

## 2. Luồng Xử lý Dữ liệu (`src/utils/data_loader.py`)

### 2.1 Tiền xử lý (`data_processing`)
1. **Chuẩn hóa**: Chia toàn bộ giá trị đặc trưng cho 255.0 để đưa về khoảng $[0, 1]$.
2. **Reshape**: Chuyển đổi dữ liệu từ dạng phẳng sang tensor 3D: `(số_mẫu, 20, 256)`.
3. **Gán nhãn**: Lấy nhãn của gói tin cuối cùng trong mỗi flow làm nhãn cho toàn bộ flow đó.

### 2.2 Chiến lược Chia dữ liệu (Semi-supervised Split)
Quy trình chia diễn ra như sau:
$$\text{Toàn bộ Target} \xrightarrow{\text{Split (0.9)}} \begin{cases} \text{Test Set (90\%)} \\ \text{Semi-supervised Set (10\%)} \xrightarrow{\text{Split (0.2)}} \begin{cases} \text{Labeled (2\% tổng)} \\ \text{Unlabeled (8\% tổng)} \end{cases} \end{cases}$$

---

## 3. Chi tiết Kiến trúc Model

### 3.1 Weak Learner: CNN Model (`src/models/cnn_model.py`)
Mỗi expert là một mạng CNN với cấu trúc:
- **Input**: `(Batch, 1, 20, 256)` (Thêm chiều channel = 1).
- **Layers**: 
    - 6 lớp Convolution (`Conv2d`) kết hợp với ReLU.
    - 3 lớp Max Pooling (`MaxPool2d`) xen kẽ để giảm chiều không gian.
- **Flatten**: Chuyển đổi feature map cuối cùng thành vector phẳng.
- **FC Layers**: 
    - `Linear(flatten_dim, 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.1)`.
    - `Linear(256, NUM_CLASSES)` $\rightarrow$ Output logits.

### 3.2 Gating Network: Mini-CNN (`src/models/gating_net.py`)
Để tối ưu khả năng chọn chuyên gia, Gating Network sử dụng kiến trúc CNN tinh gọn thay vì MLP phẳng:
- **Input**: `(Batch, 1, 20, 256)`.
- **Layers**: 
    - 2 lớp Convolution (`Conv2d`) với 16 filters $\rightarrow$ ReLU.
    - 1 lớp Max Pooling (`MaxPool2d`).
- **Attention Pooling**: Thay vì flatten hoặc GAP, mạng sử dụng cơ chế Attention để học trọng số cho từng vùng của feature map và tính tổng có trọng số. Điều này giúp Gating Network tập trung vào các vùng quan trọng nhất của traffic flow để quyết định routing.
- **Final Layer**: `Linear(16, num_learners)` $\rightarrow$ Output điểm số cho từng expert.

---

## 4. Chi tiết Training Pipeline

Quy trình huấn luyện được chia thành hai giai đoạn độc lập nhưng có sự kế thừa: xây dựng đội ngũ chuyên gia (Ensemble) và huấn luyện người điều phối (Gating Network).

### Giai đoạn 1: Huấn luyện Đội ngũ Chuyên gia (Base Ensemble)
*Mục tiêu: Tạo ra $T$ weak learners có khả năng phân loại traffic tốt trên miền đích.*

**Luồng thực hiện:**
1. **Khởi tạo**: Gán trọng số $\beta = 1/N$ cho toàn bộ mẫu dữ liệu.
2. **Vòng lặp huấn luyện tuần tự (t = 1 $\rightarrow$ T)**:
   - **Lấy mẫu (Sampling)**: Sử dụng `WeightedRandomSampler` để lấy mẫu dữ liệu dựa trên $\beta$. Mẫu có $\beta$ cao sẽ xuất hiện nhiều hơn.
   - **Huấn luyện**: Train một model CNN trên tập mẫu này $\rightarrow$ Thu được Chuyên gia $t$.
   - **Đánh giá sai số ($\epsilon_t$)**: Chạy Chuyên gia $t$ trên tập dữ liệu miền Đích để tính tỉ lệ lỗi.
   - **Tính độ tin cậy ($\alpha_t$)**: $\alpha_t = \ln((1-\epsilon_t)/\epsilon_t) + \ln(K-1)$. Chuyên gia càng ít sai thì $\alpha_t$ càng cao.
   - **Cập nhật trọng số $\beta$**:
     - **Mẫu Đích bị sai**: $\uparrow$ Tăng $\beta$ $\rightarrow$ Chuyên gia $t+1$ sẽ bị buộc phải học mẫu này.
     - **Mẫu Nguồn bị sai**: $\downarrow$ Giảm $\beta$ $\rightarrow$ Loại bỏ các tri thức từ miền nguồn gây nhiễu cho miền đích.
3. **Kết quả**: Một tập hợp $\{ (\text{CNN}_1, \alpha_1), (\text{CNN}_2, \alpha_2), \dots, (\text{CNN}_T, \alpha_T) \}$.

---

### Giai đoạn 2: Huấn luyện Người Điều phối (Gating Network)
*Mục tiêu: Học cách chọn ra Top-k chuyên gia giỏi nhất cho mỗi mẫu traffic cụ thể.*

**Luồng thực hiện bao gồm các tùy chọn (optional flags):**

#### Bước 2.1: Pre-training Bán giám sát (Optional: `--use_semi`)
*Sử dụng tập dữ liệu không nhãn (Unlabeled) để khởi tạo trọng số.*
- **Xử lý**: $\text{Dữ liệu} \xrightarrow{\text{K-Means (k=T)}} \text{Cluster IDs} \xrightarrow{\text{One-hot}} \text{Pseudo-labels}$.
- **Huấn luyện**: Gating Net học dự đoán Cluster ID $\rightarrow$ Giúp mạng hiểu đặc điểm phân phối traffic trước khi học về chuyên gia.

#### Bước 2.2: Supervised Fine-tuning
*Sử dụng nhãn Oracle để tinh chỉnh khả năng điều phối.*
- **Tạo nhãn Oracle**: 
  - **Binary Labels (Mặc định)**: Tạo vector nhị phân (1 nếu chuyên gia nằm trong Top-k đúng, 0 nếu sai).
  - **Soft Labels (Optional: `--use_soft_labels`)**: Sử dụng trọng số $\alpha_t$ chuẩn hóa: $\text{Nhãn}_{i,t} = \alpha_t / \sum \alpha_{correct}$. Giúp mạng học được mức độ tin cậy tương đối giữa các chuyên gia.
- **Huấn luyện**:
  - **Task Loss**: BCE loss giữa output của Gating Net và nhãn Oracle (Binary hoặc Soft).
  - **Balance Loss**: MSE loss để đảm bảo phân phối chuyên gia đồng đều, tránh Expert Collapse.

---

### Giai đoạn 3: Luồng Suy luận (Inference Pipeline)

Khi có một mẫu traffic mới $\mathbf{x}$:

1. **Gating**: $\mathbf{x} \rightarrow \text{Gating Network (Mini-CNN)} \rightarrow \text{Điểm số cho T experts}$.
2. **Selection**: Chọn ra Top-k experts có điểm cao nhất $\rightarrow$ $\{\text{Exp}_{i_1}, \text{Exp}_{i_2}, \dots, \text{Exp}_{i_k}\}$.
3. **Execution**: Chỉ chạy dự đoán cho $k$ experts này $\rightarrow$ $\{\hat{y}_{i_1}, \dots, \hat{y}_{i_k}\}$.
4. **Aggregation**: Kết hợp kết quả bằng bỏ phiếu có trọng số $\alpha$:
   $$\text{Kết quả} = \text{argmax} \sum_{j=1}^{k} \alpha_{i_j} \cdot \mathbb{I}(\hat{y}_{i_j} = \text{class})$$

---

## 5. Quy trình Thực thi trong `main.py`

1. **Load Data**: Gọi `load_target_data` để lấy 3 tập (Labeled, Unlabeled, Test).
2. **Phase 1 (Original)**: Chạy `model_orig.fit` $\rightarrow$ Lưu `model_orig.pth`.
3. **Phase 2 (Gating)**:
    - Khởi tạo `model_gated` với kiến trúc `GatingCNN`.
    - Gọi `train_gate` với các flag: `--use_semi` (Pre-train) và `--use_soft_labels` (Soft Oracle).
    - Lưu `model_gated.pth`.
4. **Phase 3 (Evaluation)**: 
    - So sánh Accuracy và Thời gian suy luận giữa Full Ensemble và Gated Sparse (với `GATING_K`).

---

## 6. Tối ưu hóa Phần cứng
- **CUDA**: Sử dụng `DEVICE = "cuda"`, `pin_memory=True` trong DataLoader.
- **MPS**: Sử dụng `DEVICE = "mps"`, gọi `torch.mps.synchronize()` để đo thời gian chính xác.
