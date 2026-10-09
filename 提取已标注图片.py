import os
import shutil

# ================= 配置路径 =================
# 原始图片文件夹
source_images_dir = r"I:\youyan\camera_B_20260918_112330"
# 原始标签文件夹 (YOLO格式的txt文件)
source_labels_dir = r"I:\youyan\camera_B_20260918_112330"
# 新的输出文件夹
output_dir = r"I:\youyan\yolo_dataset"

# 创建输出文件夹结构
output_images_dir = os.path.join(output_dir, "images")
output_labels_dir = os.path.join(output_dir, "labels")
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# 支持的图片格式
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# 遍历所有图片
count_copied = 0
count_skipped = 0
for img_file in os.listdir(source_images_dir):
    if not img_file.lower().endswith(image_extensions):
        continue

    img_name_without_ext = os.path.splitext(img_file)[0]
    label_file = os.path.join(source_labels_dir, img_name_without_ext + ".json")

    # 判断标签文件是否存在，且内容是否非空
    has_valid_label = False
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if content:  # 文件内容不为空
                has_valid_label = True

    if has_valid_label:
        # 复制图片和标签到新文件夹
        shutil.copy2(os.path.join(source_images_dir, img_file), os.path.join(output_images_dir, img_file))
        shutil.copy2(label_file, os.path.join(output_labels_dir, img_file.replace(os.path.splitext(img_file)[1], '.json')))
        count_copied += 1
    else:
        count_skipped += 1

print(f"处理完成！")
print(f"已复制有标注的图片: {count_copied} 张")
print(f"已跳过未标注的图片: {count_skipped} 张")
print(f"新数据集位于: {output_dir}")