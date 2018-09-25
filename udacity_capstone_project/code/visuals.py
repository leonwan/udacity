import matplotlib.pyplot as plt
import pandas as pd

def ModelLearning(file_path):
	#加载数据
	data = pd.read_csv(file_path)
	epoch_num = data['epoch']
	train_acc = data['acc']
	train_loss = data['loss']
	val_acc = data['val_acc']
	val_loss = data['val_loss']
	#创建画图窗口
	fig = plt.figure(figsize=(14,6))
	for k, name in enumerate(['accuary','loss']):
		ax = fig.add_subplot(1,2,k+1)
		if name == 'accuary':
			y_train_data = train_acc
			y_val_data = val_acc
			train_label = 'Training Accuary'
			val_label = 'Validation Accuary'
		else:
			y_train_data = train_loss
			y_val_data = val_loss
			train_label = 'Training Loss'
			val_label = 'Validation Loss'
		ax.plot(epoch_num,y_train_data,'o-',color='r',label = train_label)
		ax.plot(epoch_num,y_val_data,'o-',color='g',label = val_label)
    	#添加Labels
		ax.set_title('Learning curve of %s'%(name))
		ax.set_xlabel('epoch')
		ax.set_ylabel(name)
		#ax.set_xlim([1,epoch_num])
		ax.legend()
	fig.suptitle('Learning Performances', fontsize = 16, y = 1.03)
	fig.tight_layout()
	fig.show()
