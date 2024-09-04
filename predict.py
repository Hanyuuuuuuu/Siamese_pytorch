import os

from PIL import Image

from siamese import Siamese

import tkinter as tk
from tkinter import filedialog
'''
比较图片中目标的相似度：
1.输入目标图片和包含目标的图片
2.将两张图片输入相同的网络提取特征
3.将目标图片的特征与另图片的特征进行循环对比，计算相似度
4.框出目标
'''

if __name__ == "__main__":
    model = Siamese()


# 选择文件输入代码

        # 创建Tkinter窗口


    root = tk.Tk()
    root.withdraw()

    image_1 = input('请输入模板图片:')
    # image_1 = "./img/Angelic_01.png"
    try:
        image_1 = Image.open(image_1)
    except:
        print('Image_1 Open Error! Try again!')

    choice = input("请选择要比较的文件（F）或文件夹（D）：")

    if choice.upper() == "F":

        # 打开选择文件窗口
        file_path = filedialog.askopenfilename()
        # 读取所选文件的内容
        # 读取图片
        if file_path:
            image_2 = Image.open(file_path)
            # image_2.show()
        probability = model.detect_image(image_1, image_2)
        print(probability)
        # 输出文件内容
        # print(imgge_2)

    elif choice.upper() == "D":
        # 打开选择文件夹窗口
        folder_path = filedialog.askdirectory()
        # 定义保存图片的文件夹路径
        save_folder_path = './saved_images'
        # 判断文件夹是否存在，不存在则创建
        if not os.path.exists('save_folder_path'):
            os.makedirs('save_folder_path')
        # 循环读取文件夹中的所有图片文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # 打开图片文件
                image_2 = Image.open(os.path.join(folder_path, filename))
                # 显示图片
                # image.show()
                probability = model.detect_image(image_1, image_2)
                # 保存image_2到指定文件夹
                if probability > 0.7:
                    # 生成保存图片的文件路径
                    save_path = os.path.join('save_folder_path', filename)
                    # 保存图片
                    image_2.save(save_path)
                # # 判断probability是否大于0.7
                # if probability > 0.7:
                #     # 保存图片到output文件夹
                #     image_2.save(os.path.join('output', filename))

            # print(probability)

        # 关闭Tkinter窗口
        root.destroy()
    else:
        print("无效的选择！")

'''
    # 原输入代码
    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('Input image_2 filename:')
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        probability = model.detect_image(image_1, image_2)
        print(probability)
'''
'''  # OpenCV库打开的图像是一个numpy数组
   # input库打开的图像是一个PIL(Python Imaging Library)对象。
    while True:
        image_1 = input('请输入模板图片:')
        # image_1 = "./img/Angelic_01.png"
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('请输入待匹配图片:')
        # image_2 = "./img/Atem_01.png"
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
    # probability = model.detect_image(image_1, image_2)
    # print(probability)
'''


'''
    image_1 = cv2.imread('img/dog_184.png')
    image_2 = cv2.imread('img/dog_233.png')
  '''




