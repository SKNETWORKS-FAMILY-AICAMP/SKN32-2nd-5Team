import matplotlib.pyplot as plt
import numpy as np

# 데이터 정의
epochs = np.arange(1, 8)

# 각 모델의 f1_weighted
transformer_f1 = [0.984936, 0.989688, 0.990435, 0.992683, 0.991853, 0.989966, 0.991164]
cnn_f1 = [0.980546, 0.988922, 0.991771, 0.992317, 0.992554, 0.989549, 0.990234]
lstm_f1 = [0.843392, 0.843392, 0.974989, 0.987651, 0.99002, 0.992305, 0.993032]

# 그래프 생성
plt.figure(figsize=(10, 6))
plt.plot(epochs, transformer_f1, marker='o', linewidth=2, label='Transformer Encoder', color='#1f77b4')
plt.plot(epochs, cnn_f1, marker='s', linewidth=2, label='CNN', color='#ff7f0e')
plt.plot(epochs, lstm_f1, marker='^', linewidth=2, label='LSTM', color='#2ca02c')

# 그래프 설정
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
plt.title('F1 Score', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.xticks(epochs)
plt.ylim([0.8, 1.0])

# 그래프 저장 및 표시
plt.tight_layout()
plt.savefig('f1_score_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("그래프가 'f1_score_comparison.png'로 저장되었습니다.")
