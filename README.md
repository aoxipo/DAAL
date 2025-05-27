
<div align="center">
  <div>
    <h1>
        Dynamically adaptive adjustment loss function biased towards few class learning
    </h1>
  </div>
  <div>
      <a href='https://www.htu.edu.cn/cs/2018/0524/c10537a120622/page.htm'>Guoqi Liu</a> &emsp; 
      <a href='https://ietresearch.onlinelibrary.wiley.com/authored-by/Bai/Lu'>Lu Bai</a> &emsp; 
      <a >Junlin Li</a> &emsp;
      <a href='roolenyuan@163.com'>Linyuan Ru*</a> &emsp;
      <a href='https://www.htu.edu.cn/cs/2019/0227/c10537a138434/page.htm'>Baofang Chang</a>
  </div>
  <br/>
</div>

DAAL is  adaptive adjustment loss function based on the paper "Dynamically adaptive adjustment loss function biased towards few class learning", which has been accepted by IET 2022.

 

## Framework

DAAL aims to address challenges related to class imbalance, a key issue in computer vision tasks.  To this end, DAAL strategically balances the influence of majority and minority classes through a dual-pronged approach.  Inspired by the need to ensure fair learning across all classes, DAAL first introduces batch nuclear-norm maximization to promote diverse and discriminative feature learning, followed by the establishment of an adaptive composite loss function to dynamically adjust the loss contributions of different classes, thereby enhancing the prediction accuracy and interpretability of the deep neural network framework.

## Method
We achieve the optimal state estimation based on the dynamic estimation of Kalman filtering and the batch kernel norm maximization loss by predicting the proportionally update cycle of the combination of loss functions., formulated as:
 
#### 1. Prediction Step
**State Prediction**:  
$\hat{x}_{k|k-1} = F_k \hat{x}_{k-1|k-1} + B_k u_k$  
- $\hat{x}_{k|k-1}$: Predicted state estimate at time $k$  
- $F_k$: State transition matrix (system dynamics)  
- $B_k$: Control matrix, $u_k$: Control input (optional)
 
**Covariance Prediction**:  
$P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k$  
- $P_{k|k-1}$: Prediction uncertainty covariance  
- $Q_k$: Process noise covariance (model imperfections)
 
#### 2. Update Step
**Kalman Gain Calculation**:  
$K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R_k)^{-1}$  
- $K_k$: Kalman gain (balances prediction vs measurement)  
- $H_k$: Measurement matrix (maps state to observation space)  
- $R_k$: Measurement noise covariance (sensor noise)
 
**State Update**:  
$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - H_k \hat{x}_{k|k-1})$  
- $z_k$: Actual measurement at time $k$  
- $\hat{x}_{k|k}$: Optimal state estimate after fusion
 
**Covariance Update**:  
$P_{k|k} = (I - K_k H_k) P_{k|k-1}$  
- Updates uncertainty for next iteration
 
### Kalman Adaptive and Batch Nuclear-norm Maximization Loss Function
 
#### 1. Adaptive Dynamic Loss (Time Series Forecasting)
**Concept**: Use Kalman Gain $K_k$ to dynamically weight prediction vs ground truth losses  
**Formula**:  
$L = \sum_{k=1}^T \left[ K_k \cdot \text{MSE}(y_k, \hat{y}_k) + (1 - K_k) \cdot \text{MSE}(y_{k-1}, \hat{y}_{k|k-1}) \right]$  
- $y_k$: Ground truth at time $k$  
- $\hat{y}_{k|k-1}$: Prior estimate from Kalman prediction  
- High $K_k$ (measurement trust) when low process noise
 
#### 2. Uncertainty-Aware Loss (Object Detection)
**Concept**: Penalize high-uncertainty predictions less using covariance matrix $P_k$  
**Formula**:  
$L_{reg} = \sum_{i=1}^N \frac{1}{\sqrt{\det(P_k^{(i)})}} \cdot \text{SmoothL1}(\Delta x_i, \Delta \hat{x}_i)$  
- $\det(P_k^{(i)})$: Prediction uncertainty for object $i$  
- Reduces loss weight for uncertain detections
 
### Implementation Snippet
```python
# Kalman-inspired dynamic loss function
class KalmanLoss(nn.Module):
    def __init__(self, state_dim, F, H, Q, R):
        super().__init__()
        self.F = F  # State transition matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = torch.eye(state_dim)  # Initial uncertainty
 
    def forward(self, pred, true, control=None):
        # Prediction step
        x_pred = self.F @ pred + self.B @ control  # State propagation
        P_pred = self.F @ self.P @ self.F.T + self.Q  # Uncertainty propagation
 
        # Kalman gain calculation
        S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
        K = P_pred @ self.H.T @ torch.inverse(S)  # Optimal gain
 
        # Update step
        y = true - self.H @ x_pred  # Measurement residual
        x_est = x_pred + K @ y  # Optimal state estimate
        I = torch.eye(self.P.shape[0]).to(K.device)
        self.P = (I - K @ self.H) @ P_pred  # Update uncertainty
 
        # Dynamic loss weighting (example: use trace of K)
        loss_weight = K.diag().mean()
        return F.l1_loss(x_est, true) * loss_weight
```
## Loss   

The loss of the four experimental schemes changes with theincrease of the epochs

<img src="https://github.com/aoxipo/DAAL/blob/main/image/loss.png" style="zoom: 67%;" />

## Result 

here is our result for object detection, crowd counting task

<img src="https://github.com/aoxipo/DAAL/blob/main/image/Result.png" alt="Overview" style="zoom:80%;" />



## Citation

Please cite this work if you find it useful:

```latex
@article{https://doi.org/10.1049/ipr2.12661,
author = {Liu, Guoqi and Bai, Lu and Li, Junlin and Li, Xusheng and Ru, Linyuan and Chang, Baofang},
title = {Dynamically adaptive adjustment loss function biased towards few-class learning},
journal = {IET Image Processing},
volume = {17},
number = {2},
pages = {627-635},
doi = {https://doi.org/10.1049/ipr2.12661},
url = {https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/ipr2.12661},
eprint = {https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/ipr2.12661},
abstract = {Abstract Convolution neural networks have been widely used in the field of computer vision, which effectively solve practical problems. However, the loss function with fixed parameters will affect the training efficiency and even lead to poor prediction accuracy. In particular, when there is a class imbalance in the data, the final result tends to favor the large-class. In detection and recognition problems, the large-class will dominate due to its quantitative advantage, and the features of few-class can be not fully learned. In order to learn few-class, batch nuclear-norm maximization is introduced to the deep neural networks, and the mechanism of the adaptive composite loss function is established to increase the diversity of the network and thus improve the accuracy of prediction. The proposed loss function is added to the crowd counting, and verified on ShanghaiTech and UCF\_CC\_50 datasets. Experimental results show that the proposed loss function improves the prediction accuracy and convergence speed of deep neural networks.},
year = {2023}
}
```

## Contact

If you have any specific questions or if there's anything else you'd like assistance with regarding the code, feel free to let me know. 
