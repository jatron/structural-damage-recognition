import numpy as np
import csv

iv3 = np.loadtxt('scores_inceptionv3.txt')
mb1 = np.loadtxt('scores_mobile1.txt')

with open('submission_merge.csv','w') as csvfile:
  fieldnames=['Index','Pred']
  writer = csv.DictWriter(csvfile,delimiter=',',fieldnames=fieldnames)
  writer.writeheader()
  for i in range(len(iv3)):
    if iv3[i,2] >= mb1[i,2]:
       writer.writerow({'Index':str(i),'Pred':str(int(iv3[i,1]))})
    else:
       writer.writerow({'Index':str(i),'Pred':str(int(mb1[i,1]))})

