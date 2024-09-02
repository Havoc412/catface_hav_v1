import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def count_images(parent_folder):
    """
    统计指定父文件夹下每个子文件夹中的图片数量。
    """
    folder_dict = {}
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            count = 0
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    count += 1
            folder_dict[folder] = count
    return folder_dict


def show_images(folder_path):
    """
    展示指定文件夹下的所有图片。
    """
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
             f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    num_files = len(files)
    if num_files == 0:
        print("没有找到图片。")
        return

    cols = 3  # 可调整每行显示的图片数量
    rows = (num_files + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    ax = ax.ravel() if num_files > 1 else [ax]

    for i, file in enumerate(files):
        img = mpimg.imread(file)
        ax[i].imshow(img)
        ax[i].set_title(os.path.basename(file))
        ax[i].axis('off')

    for i in range(num_files, len(ax)):
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()


# 示例使用
parent_folder = 'path_to_your_parent_folder'
counts = count_images(parent_folder)
print("文件夹中的图片数量：")
for folder, count in counts.items():
    print(f"{folder}: {count}张图片")

# 展示特定子文件夹中的图片
folder_to_show = 'path_to_specific_folder'
show_images(folder_to_show)