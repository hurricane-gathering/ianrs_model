import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Attention, Flatten, Dense,GlobalAveragePooling2D, Dropout, Input,BatchNormalization,MultiHeadAttention, LayerNormalization, Add

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet import ResNet50

# 限制CPU使用
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
 



# 读取CSV文件，包含图像名称和所属分组
data = pd.read_csv('data/train.csv')
classt = 'group'
# 划分数据集为训练集和验证集
train_data, val_data = train_test_split(data, test_size=0.8, stratify=data[classt])

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
    directory='data/train',  # 图像所在的目录路径
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
    directory='data/train',
    x_col='name',
    y_col=classt,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # 在验证集上不需要打乱数据
)


# 创建MuNet模型
# def create_munet(input_shape):
#     # 输入层
#     inputs = Input(shape=input_shape)
    
#     # 卷积层和池化层
#     x = Conv2D(32, (3, 3), activation='relu')(inputs)
#     x = MaxPooling2D((2, 2))(x)
#     x = Conv2D(64, (3, 3), activation='relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = Conv2D(128, (3, 3), activation='relu')(x)
#     x = MaxPooling2D((2, 2))(x)
    
#     # 全连接层
#     # x = Dropout(0.25)(x)
#     x = Flatten()(x)
#     x = Dense(256, activation='relu')(x)
#     x = Dropout(0.25)(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.25)(x)
#     # x = Dense(64, activation='relu')(x)
#     # x = Dropout(0.5)(x)
    
#     # 输出层
#     outputs = Dense(10, activation='soft    max')(x)  # 多分类问题的输出层
    
#     model = Model(inputs, outputs)
#     return model

def create_munet(input_shape):
    # 输入层
    inputs = Input(shape=input_shape)
    
    # 卷积层和池化层
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

     # 注意力机制
    query = Dense(128)(x)
    key = Dense(128)(x)
    value = Dense(128)(x)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=128)(query, key, value)
    attention_output = Add()([x, attention_output])
    attention_output = LayerNormalization()(attention_output)

    # Transformer层
    transformer_output = MultiHeadAttention(num_heads=4, key_dim=128)(attention_output, attention_output)
    transformer_output = Add()([attention_output, transformer_output])
    transformer_output = LayerNormalization()(transformer_output)

    # 全局平均池化层
    x = GlobalAveragePooling2D()(transformer_output)
    
    # 全连接层
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # 输出层
    outputs = Dense(10, activation='softmax')(x)  # 多分类问题的输出层
    
    model = Model(inputs, outputs)
    return model

# 定义模型输入形状
input_shape = (image_width, image_height, 3)  # 根据数据输入形状进行调整

# 创建MuNet模型                                                                                                         
model = create_munet(input_shape)
# 加载 ResNet-RS-350 模型
# model = ResNet50(weights='imagenet')  # 加载预训练权重

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # 分类问题使用交叉熵损失
              metrics=['accuracy'])

# 打印模型结构
model.summary()

# 定义 EarlyStopping 回调
early_stopping = EarlyStopping(monitor='val_accuracy',  # 监控验证集的精度
                               patience=300,  # 10个epoch内精度没有提升则停止训练
                               restore_best_weights=True)  # 恢复最佳模型权重


# 训练模型
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=1000,
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    callbacks=[early_stopping])

history_dict = history.history


# 保存模型
model.save("results/tfJD.keras")

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

f.savefig("results/tfJD.png", bbox_inches='tight')

# plt.show()
# 评估模型
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(f'Test loss: {test_loss},Test accuracy: {test_acc}')