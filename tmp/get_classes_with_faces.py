import os
import shutil

folder_path = "..\\dataface\\train\\"
new_folder = "..\\dataface\\train_face\\"

count = 0

for dir_list, subdir_list, file_name in os.walk(folder_path):
    for person in subdir_list:
        face_dir_path = dir_list + person + '\\face\\'
        if not os.path.exists(face_dir_path):
            count = count + 1
            continue
        for idx, fname in enumerate(os.listdir(face_dir_path)):
            img_path = face_dir_path + fname
            if idx <= 9:
                newimg_name = person + '_000' + str(idx) + '.jpg'
            elif idx <= 99:
                newimg_name = person + '_00' + str(idx) + '.jpg'
            elif idx <= 999:
                newimg_name = person + '_0' + str(idx) + '.jpg'
            else:
                newimg_name = person + '_' + str(idx) + '.jpg'

            newsubdir_path = new_folder + person
            if not os.path.exists(newsubdir_path):
                os.mkdir(newsubdir_path)
            newimg_path = newsubdir_path + '\\' + newimg_name
            shutil.copy(img_path, newimg_path)
            # print(img_path)
            # print(newimg_path)
    break

print('Number of classes without faces: ', count)
