# import pandas as pd
# import json
#
#
# # 读取Excel文件
# def read_excel(file_path):
#     return pd.read_excel(file_path, engine='openpyxl')
#
#
# # 将评论转换为JSON格式
# def convert_review_to_json(comment):
#     return {"text": comment}
#
#
# def main():
#     file_path = 'C:/Users/ASUS/Desktop/product_reviews.xlsx'  # Excel文件路径
#     reviews_df = read_excel(file_path)  # 读取Excel文件到DataFrame
#
#     # 创建包含所有评论的JSON数组
#     reviews_json_list = []
#
#     for index, row in reviews_df.iterrows():
#         comment = row["Comment"]
#
#         # 跳过空评论
#         if pd.isna(row["Comment"]):
#             continue
#
#         # 获取每一条评论并转换为JSON格式
#         comment_json_data = convert_review_to_json(comment)
#         reviews_json_list.append(comment_json_data)
#
#     # 输出文件的路径
#     json_output_path = 'C:/Users/ASUS/Desktop/EventExtraction-for-Productreviews-main/datashop/data.json'
#
#     # 写入标准的Json文件
#     with open(json_output_path, 'w', encoding='utf-8') as json_file:
#         json.dump(reviews_json_list, json_file, ensure_ascii=False, indent=2)
#
#
# if __name__ == "__main__":
#     main()

import pandas as pd
import json


# 读取Excel文件
def read_excel(file_path):
    return pd.read_excel(file_path, engine='openpyxl')


# 将评论转换为JSON格式
def convert_review_to_json(comment):
    # 注意这里没有额外的冒号
    return {"text": comment}


def main():
    file_path = 'C:/Users/ASUS/Desktop/product_reviews.xlsx'  # Excel文件路径
    reviews_df = read_excel(file_path)  # 读取Excel文件到DataFrame

    # 输出文件的路径
    json_output_path = 'C:/Users/ASUS/Desktop/EventExtraction-for-Productreviews-main/datashop/data.json'

    # 打开输出文件准备写入数据
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        for index, row in reviews_df.iterrows():
            comment = row["Comment"]

            # 跳过空评论
            if pd.isna(row["Comment"]):
                continue

            # 获取每一条评论并转换为JSON格式的字符串
            comment_json_str = json.dumps(convert_review_to_json(comment), ensure_ascii=False)

            # 写入JSON字符串并加上换行符
            json_file.write(comment_json_str + '\n')


if __name__ == "__main__":
    main()