import matplotlib.pyplot as plt
import numpy as np

# 데이터 정의
epochs = np.arange(1, 8)

# 각 모델의 valid_loss
transformer_valid_loss = [0.013992, 0.011112, 0.011422, 0.010355, 0.010115, 0.013634, 0.018798]
cnn_valid_loss = [0.025665, 0.016881, 0.024703, 0.045062, 0.035248, 0.047737, 0.07478]
lstm_valid_loss = [0.092761, 0.081839, 0.029777, 0.014458, 0.017815, 0.010689, 0.011789]

# 그래프 생성
plt.figure(figsize=(10, 6))
plt.plot(epochs, transformer_valid_loss, marker='o', linewidth=2, label='Transformer Encoder', color='#1f77b4')
plt.plot(epochs, cnn_valid_loss, marker='s', linewidth=2, label='CNN', color='#ff7f0e')
plt.plot(epochs, lstm_valid_loss, marker='^', linewidth=2, label='LSTM', color='#2ca02c')

# 그래프 설정
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Valid Loss', fontsize=12, fontweight='bold')
plt.title('Model Comparison: Valid Loss by Epoch', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.xticks(epochs)

# 그래프 저장 및 표시
plt.tight_layout()
plt.savefig('model_valid_loss_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("그래프가 'model_valid_loss_comparison.png'로 저장되었습니다.")
