import torch

def compare_torch_objects(path1, path2):
    obj1 = torch.load(path1, map_location='cpu')
    obj2 = torch.load(path2, map_location='cpu')

    if type(obj1) != type(obj2):
        return False

    # 만약 state_dict 또는 dict 타입일 경우
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        if obj1.keys() != obj2.keys():
            return False
        for k in obj1.keys():
            v1 = obj1[k]
            v2 = obj2[k]
            if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                if not torch.equal(v1, v2):
                    return False
            else:
                if v1 != v2:
                    return False
        return True

    # 만약 Tensor이면
    if isinstance(obj1, torch.Tensor) and isinstance(obj2, torch.Tensor):
        return torch.equal(obj1, obj2)

    # 기타 타입은 그냥 ==
    return obj1 == obj2

# 사용 예:
print(compare_torch_objects("act_scales/Llama-2-7b-hf-fuquant.pt", "act_scales/Llama-2-7b-hf.pt"))
