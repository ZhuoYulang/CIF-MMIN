miss_shared_dict = {
    "azz": []
}

miss_shared_dicts = []

for i in range(0, 3):
    miss_shared_dicts.append({
        "azz": [],
        "zvz": [],
        "zzl": [],
        "avz": [],
        "azl": [],
        "zvl": [],
    })

for i in range(0, 3):
    miss_shared_dicts[i]["azz"].append('1')
    print(i, miss_shared_dicts[i]["azz"])

print(miss_shared_dicts)
