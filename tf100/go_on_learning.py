from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

model = load_model('results/tf100.keras')  # 加载模型


# 读取CSV文件，包含图像名称和所属分组
data = pd.read_csv('../data/train_demo/train100.csv')
classt = 'group'
# 划分数据集为训练集和验证集
train_data, val_data = train_test_split(data, test_size=0.6, stratify=data[classt])

# 将整数标签转换为字符串标签
train_data[classt] = train_data[classt].astype(str)
val_data[classt] = val_data[classt].astype(str)

# 数据生成器设置
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # 对图像进行归一化
    rotation_range=20,  # 随机旋转角度
    width_shift_range=0.2,  # 水平随机平移
    height_shift_range=0.2,  # 垂直随机平移
    horizontal_flip=True,  # 水平翻转
    zoom_range=0.1  # 随机缩放
)

batch_size = 8
image_height = 300
image_width = 300

# 训练数据生成器
train_generator = datagen.flow_from_dataframe(
    train_data,
    directory='../data/train_demo/train100',  # 图像所在的目录路径
    x_col='name',  # 图像文件名列
    y_col=classt,  # 标签列
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',  # 多分类任务
    shuffle=True  # 随机打乱数据
)

# 验证数据生成器
val_generator = datagen.flow_from_dataframe(
    val_data,
    directory='../data/train_demo/train100',
    x_col='name',
    y_col=classt,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # 在验证集上不需要打乱数据
)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型结构
model.summary()

# 定义 EarlyStopping 回调
early_stopping = EarlyStopping(monitor='val_accuracy',  # 监控验证集的精度
                               patience=10,  # 10个epoch内精度没有提升则停止训练
                               restore_best_weights=True)  # 恢复最佳模型权重
# 继续训练模型
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=100,
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    callbacks=[early_stopping])

history_dict = history.history


# 保存模型
model.save("results/tf100_go_on_learning.keras")

# 绘制训练曲线
f = plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, label='Training loss')

plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# 调整子图之间的间距
plt.tight_layout()

f.savefig("results/tf100_go_on_learning.png", bbox_inches='tight')