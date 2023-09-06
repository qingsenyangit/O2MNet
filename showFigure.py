import matplotlib.pyplot as plt


train_val_info_dir = './trained-model/lossTXT.txt'
# train_val_info_dir = './trained-model/lossTXT.txt'
# test_info_dir = 'test_info.txt'
train_val_info = open(train_val_info_dir).readlines()
# test_info = open(test_info_dir).readlines()

epoch = [int(f.split(' iteration')[0].split('Epoch ')[-1]) for f in train_val_info]
train_acc = [float(f.split('loss: ')[-1]) for f in train_val_info]
# val_acc = [float(f.split('loss: ')[-1]) for f in train_val_info]

# epoch_test = [int(f.split('trained_model')[1].split('.')[0]) for f in test_info]
# test_acc = [float(f.split('train_acc:')[1].split(',')[0][:-1]) for f in test_info]

# epoch1 = list(range(5194))
plt.figure()
plt.plot(epoch, train_acc, label = 'train_acc')
# plt.plot(epoch1, val_acc, label = 'val_acc')
# plt.plot(epoch_test, test_acc, label = 'test_acc')
plt.plot(epoch, train_acc, 'g*')
# plt.plot(epoch1, val_acc, 'b*')
# plt.plot(epoch_test, test_acc, 'r*')
plt.grid(True)
plt.legend(loc=4)
plt.axis('tight')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()
