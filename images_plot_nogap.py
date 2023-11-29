import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# 이미지 파일명 리스트
# 정상
image_files = ['output_9.png', 'output_11.png', 
               'output_13.png', 'output_15.png', 
                'output_18.png', 'output_19.png']

# 비정상
image_files = ['output_16.png', 
                'output_17.png', 'output_20.png']

# 이미지 불러오기
import os
images = [Image.open(os.path.join('result_images',file)) for file in image_files]

# plot 설정
fig = plt.figure(figsize=(11, 7))
gs = gridspec.GridSpec(1, 3, hspace=0.01, wspace=0.01)

# 이미지를 간격 좁게 표시
for i in range(3):
    ax = fig.add_subplot(gs[i])
    ax.imshow(images[i])
    ax.axis('off')

# 결과 출력
plt.show()
