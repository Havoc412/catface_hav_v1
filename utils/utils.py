import math


def distance(pt1, pt2):
    res = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    return res


def cal_degree(pt1, pt2):
    """ 计算偏移角度 """
    tan_value = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0] + 1e-10)  # tip 增加一个浮点数，防止溢出

    radians = math.atan(tan_value)
    degrees = math.degrees(radians)

    return degrees


def merge_breeds(breeds, proportion=None, eps=0.01):
    """

    :param breeds:
    :param proportion: conf 的 二级比重，默认都为 1
    :param eps: conf 的最小值，太就没有数目参考价值了。
    :return:
    """
    # 初始化字典来存储累加的 conf 值和计数
    conf_accumulator = {}
    # 累加 conf 值
    for breed in breeds:
        for top, conf in zip(breed['top5'], breed['conf']):
            if top in conf_accumulator:
                conf_accumulator[top]['total_conf'] += conf
            else:
                conf_accumulator[top] = {'total_conf': conf}
    # 重排
    sorted_conf = sorted(conf_accumulator.items(), key=lambda item: item[1]['total_conf'], reverse=True)
    # 提取排序后的 top5 和 conf
    if proportion is None:
        proportion = len(breeds)
    sorted_conf_values = [item[1]['total_conf'] / proportion for item in sorted_conf
                          if item[1]['total_conf'] / proportion > eps]  # 同时实现过滤概率过低的目标，降低后续的计算量。
    sorted_top5 = [item[0] for idx, item in enumerate(sorted_conf)
                   if idx < len(sorted_conf_values)]
    return {
        'top5': sorted_top5,
        'conf': sorted_conf_values
    }


if __name__ == "__main__":
    all_breeds = [
        {'top5': ['A', 'B', 'C'], 'conf': [0.2, 0.3, 0.5]},
        {'top5': ['A', 'D', 'E'], 'conf': [0.1, 0.4, 0.5]},
        {'top5': ['B', 'C', 'D'], 'conf': [0.3, 0.3, 0.4]}
    ]
    print(merge_breeds(all_breeds))



