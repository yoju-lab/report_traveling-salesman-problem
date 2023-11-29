import matplotlib.pyplot as plt

# 이미지 파일명 리스트
image_files = ['output_5.png', 'output_6.png', 'output_7.png', 'output_8.png',
               'output_9.png', 'output_10.png', 'output_11.png', 'output_12.png',
               'output_13.png', 'output_14.png', 'output_15.png', 'output_16.png',
               'output_17.png', 'output_18.png', 'output_19.png', 'output_20.png']

# plot에 이미지 배치
fig, axes = plt.subplots(4, 4, figsize=(14, 8))

import os
for i, ax in enumerate(axes.flat):
    images_path = os.path.join('result_images',image_files[i])
    img = plt.imread(images_path)
    ax.imshow(img)
    ax.axis('off')

# 결과 출력
plt.tight_layout()
plt.show()
pass
