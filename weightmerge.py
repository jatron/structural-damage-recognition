import numpy as np
import csv

iv3 = np.loadtxt('scores_inceptionv3.txt')
mb1 = np.loadtxt('scores_mobile1.txt')

with open('submission_weightmerge.csv','w') as csvfile:
  fieldnames=['Index','Pred']
  writer = csv.DictWriter(csvfile,delimiter=',',fieldnames=fieldnames)
  writer.writeheader()
  for i in range(len(iv3)):
    piv3 = iv3[i,2]
    pmb1 = mb1[i,2]
    if np.random.uniform(0,1) <= piv3/(piv3+pmb1):
       writer.writerow({'Index':str(i),'Pred':str(int(iv3[i,1]))})
    else:
       writer.writerow({'Index':str(i),'Pred':str(int(mb1[i,1]))})

