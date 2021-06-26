import json

data = {
    'no': 1,
    'name': "Runoob",
    'url': "http://www.runoob.com"
}

json_str = json.dumps(data)
print("python 原始数据： ", repr(data))
print("JSON 对象：", json_str)

list1 = [3,4,55,6]

### 写入json文件
with open("data.json", 'w') as f:
    # json.dump(data, f)
    json.dump(list1, f)

with open("data.json", 'r') as f:
    list2 = json.load(f)
    print(type(list2))
    print(list2[3])


