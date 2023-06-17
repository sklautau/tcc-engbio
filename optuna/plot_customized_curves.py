# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:27:57 2023

@author: sofia2022

Example: run
(cuda11.2) python plot_from_history.py --root_folder=..\optuna_outputs_2
to obtain the values
"""
import matplotlib.pyplot as plt


x1=[1, 2, 3, 4, 5, 6, 7, 8, 9]
y1=[1.3783518075942993, 0.6791130900382996, 0.6319942474365234, 0.966278076171875, 0.5834797620773315, 0.6104251146316528, 3.9612555503845215, 1.5479216575622559, 0.43670445680618286]
x2=[1, 2, 3, 4, 5, 6, 7, 8]
y2=[1.4397004842758179, 1.3118304014205933, 1.4801925420761108, 2.36093807220459, 1.9374005794525146, 2.042585611343384, 2.6481435298919678, 2.559536933898926]
y3=[0.9456117749214172, 0.6845609545707703, 0.6903135180473328, 0.7329459190368652, 0.8495065569877625, 0.5805801153182983, 1.2174855470657349, 0.7324314117431641, 0.6876479983329773]
y4=[0.680278480052948, 0.7831092476844788, 2.8629958629608154, 2.727368116378784, 0.8170190453529358, 7.261852264404297, 8.198395729064941, 13.580853462219238]
plt.plot(x1, y1, '-o', x2, y2, '-x', x1, y3, '--o', x2, y4, '--x')
plt.xlabel('# epoch step')
plt.ylabel('Loss')
plt.legend(["Trial 0 - train loss","Trial 11  - train loss", "Trial 0 - val loss","Trial 11  - val loss"])
plt.savefig("optune_study_2.png")
plt.show()


# Accuracy
x= [100, 150, 200, 250, 300, 30, 350, 400, 450, 500, 50, 550, 600, 648]
y1= [0.6820651888847351, 0.6875, 0.70923912525177, 0.7445651888847351, 0.73097825050354, 0.510869562625885, 0.70652174949646, 0.741847813129425, 0.7554348111152649, 0.741847813129425, 0.5570651888847351, 0.7527173757553101, 0.7364130616188049, 0.7635869383811951]

y2= [0.70652174949646, 0.741847813129425, 0.7445651888847351, 0.7527173757553101, 0.7554348111152649, 0.66847825050354, 0.77173912525177, 0.7744565010070801, 0.758152186870575, 0.75, 0.679347813129425, 0.760869562625885, 0.758152186870575, 0.77173912525177]
plt.plot(x, y1, 'o', x, y2, 'x')
plt.xlabel('number of training examples')
plt.ylabel('Accuracy')
plt.legend(["trained with AUC","trained with Accuracy"])
plt.show()

#Precision
y1= [0.6401673640167364, 0.648068669527897, 0.7103825136612022, 0.7205882352941176, 0.7707006369426752, 0.5059171597633136, 0.6727272727272727, 0.7379679144385026, 0.7901234567901234, 0.7986577181208053, 0.5408560311284046, 0.7183098591549296, 0.7543859649122807, 0.7437185929648241]

y2= [0.6792452830188679, 0.7129186602870813, 0.7045454545454546, 0.743455497382199, 0.7373737373737373, 0.7313432835820896, 0.7525252525252525, 0.7853107344632768, 0.7486910994764397, 0.7169811320754716, 0.6793478260869565, 0.7424242424242424, 0.7317073170731707, 0.8164556962025317]
plt.plot(x, y1, 'o', x, y2, 'x')
plt.xlabel('number of training examples')
plt.ylabel('Precision')
plt.legend(["trained with AUC","trained with Accuracy"])
plt.show()

# Recall
y1= [0.8315217391304348, 0.8206521739130435, 0.7065217391304348, 0.7989130434782609, 0.657608695652174, 0.9293478260869565, 0.8043478260869565, 0.75, 0.6956521739130435, 0.6467391304347826, 0.7554347826086957, 0.8315217391304348, 0.7010869565217391, 0.8043478260869565]

y2= [0.782608695652174, 0.8097826086956522, 0.842391304347826, 0.7717391304347826, 0.7934782608695652, 0.532608695652174, 0.8097826086956522, 0.7554347826086957, 0.7771739130434783, 0.8260869565217391, 0.6793478260869565, 0.7989130434782609, 0.8152173913043478, 0.7010869565217391]
plt.plot(x, y1, 'o', x, y2, 'x')
plt.xlabel('number of training examples')
plt.ylabel('Recall')
plt.legend(["trained with AUC","trained with Accuracy"])
plt.show()
