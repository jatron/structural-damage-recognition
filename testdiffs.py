import numpy as np
from PIL import Image

iv3 = np.loadtxt('scores_inceptionv3.txt')
mb1 = np.loadtxt('scores_mobile1.txt')

for i in range(len(iv3)):
   if iv3[i,1] != mb1[i,1]:
     print(i)
     im = Image.open('test/testimg'+str(i)+'.jpg')
     im.show()
