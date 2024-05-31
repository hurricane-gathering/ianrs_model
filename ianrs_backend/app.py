from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from PIL import Image
import random

# 初始化 Flask 应用
app = Flask(__name__)

# 加载训练好的模型
model = load_model('../tf100/results/tf100.keras')

# 定义模型输入尺寸（根据你的模型需要进行修改）
input_shape = (300, 300, 3)

# 定义模型分类标签（根据你的模型需要进行修改）
labels = ["卫衣",
    "小羽绒服",
    "大羽绒服",
    "大白褂",
    "小棉袄",
    "大棉袄",
    "冲锋衣",
    "保暖衣",
    "长款羽绒服",
    "T恤"]

# 定义预处理函数（根据你的模型需要进行修改）
def preprocess_image(image):
    # 将字节数据转换为图像对象
    img = Image.open(image)
    
    # 对图像进行随机旋转
    # rotate_angle = random.randint(-20, 20)
    # img = img.rotate(rotate_angle)
    
    # # 对图像进行随机平移
    # width_shift = random.uniform(-0.2, 0.2) * img.width
    # height_shift = random.uniform(-0.2, 0.2) * img.height
    # img = img.transform(img.size, Image.AFFINE, (1, 0, width_shift, 0, 1, height_shift))
    
    # # 对图像进行水平翻转
    # if random.random() < 0.5:
    #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # # 对图像进行随机缩放
    # zoom_factor = random.uniform(0.9, 1.1)
    # img = img.resize((int(img.width * zoom_factor), int(img.height * zoom_factor)))
                                                                                                                        
    # 调整图像尺寸
    img = img.resize((input_shape[0], input_shape[1]))
    
    # 将图像转换为数组并进行归一化处理
    img_array = np.array(img) / 255.0
    return img_array

# 定义路由，处理 POST 请求
@app.route('/classify', methods=['POST'])
def classify():
    # 获取 POST 请求中的图像数据
    image = request.files['image']
    
    # 对图像进行预处理
    preprocessed_image = preprocess_image(image)
    
    # 将图像转换为 Numpy 数组，并调整尺寸以适应模型输入
    img_array = np.reshape(preprocessed_image, (1, *input_shape))
    
    # 使用模型进行分类
    predictions = model.predict(img_array)
    
    # 获取前三个预测结果及对应的概率
    top3_indices = np.argsort(predictions)[0, -3:][::-1]  # 获取概率最高的前三个类别的索引
    top3_labels = [labels[idx] for idx in top3_indices]
    top3_probabilities = [float(predictions[0, idx]) for idx in top3_indices]
    
    # 构建返回的 JSON 数据
    result = {
        'predictions': [{
            'label': label,
            'probability': prob
        } for label, prob in zip(top3_labels, top3_probabilities)]
    }
    
    # 返回 JSON 格式的分类结果
    return jsonify(result)

# 启动 Flask 应用
if __name__ == '__main__':
    app.run(debug=True)
