import os
import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로
csv_path = os.path.join('preprocessing', 'output', 'bert.csv')

# CSV 읽기
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f'파일을 찾을 수 없습니다: {csv_path}')

# 필요한 열 검사
if 'epoch' not in df.columns or 'f1_weighted' not in df.columns:
    raise ValueError('bert.csv에 epoch 또는 f1_weighted 열이 없습니다.')

# 숫자형 epoch만 사용
df = df[pd.to_numeric(df['epoch'], errors='coerce').notna()].copy()
df['epoch'] = df['epoch'].astype(int)

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['f1_weighted'], marker='o', linewidth=2, color='#1f77b4')

plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
plt.title('KLUE-BERT', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(df['epoch'])

plt.tight_layout()
output_path = 'bert_f1_score.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"그래프가 '{output_path}'로 저장되었습니다.")
