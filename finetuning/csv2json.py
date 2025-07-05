import os.path

import pandas as pd
import json

def get_prompt(num):

    title_optimize_prompt = f"""输入的内容是一篇文档中所有标题组成的图片，共{num}行，请给每行标题确认对应的层级，使结果符合正常文档的层次结构。
    注意：
    1、为每个标题元素添加适当的层次结构
    2、行高较大或字体越浓的标题一般是更高级别的标题
    3、标题从前至后的层级必须是连续的，不能跳过层级
    4、标题层级最多为4级，不要添加过多的层级
    5、优化后的标题只保留代表该标题的层级的整数，不要保留其他信息
    6、字典中可能包含被误当成标题的正文，你可以通过将其层级标记为 0 来排除它们
    """
    return title_optimize_prompt

out_path = r"D:\CCKS2025\data\_traindata"
csv_path = r"D:\CCKS2025\data\_traindata\title_train.csv"
train_json_path = os.path.join(out_path, 'title_train.json')
val_json_path = os.path.join(out_path, 'title_val.json')
df = pd.read_csv(csv_path)
# Create conversation format
conversations = []

# Add image conversations
for i in range(len(df)):

    json_str = df.iloc[i]['text'].replace("'", '"')

    # 2. 解析为Python字典
    data_dict = json.loads(json_str)

    prompt = get_prompt(len(data_dict))

    conversations.append({
        "id": f"identity_{i+1}",
        "conversations": [
            {
                "role": "user",
                "value": f"{df.iloc[i]['image_path']}\n{prompt}"
            },
            {
                "role": "assistant",
                "value": str(df.iloc[i]['text'])
            }
        ]
    })

# print(conversations)
# Save to JSON
# Split into train and validation sets
train_conversations = conversations[:-4]
val_conversations = conversations[-4:]

# Save train set
with open(train_json_path, 'w', encoding='utf-8') as f:
    json.dump(train_conversations, f, ensure_ascii=False, indent=2)

# Save validation set
with open(val_json_path, 'w', encoding='utf-8') as f:
    json.dump(val_conversations, f, ensure_ascii=False, indent=2)