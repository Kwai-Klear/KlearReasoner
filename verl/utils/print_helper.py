def print_nested_dict_as_table(data):
    """
    将一个嵌套字典（外层key为stage，内层key为metric）以表格形式打印。

    Args:
        data (dict): 输入的嵌套字典，格式为 {stage_name: {metric_name: metric_value}}
    """
    if not data:
        print("字典为空，无法打印表格。")
        return

    # 1. 收集所有唯一的metric名称作为列头
    metric_names = set()
    for stage_data in data.values():
        metric_names.update(stage_data.keys())
    metric_names = sorted(list(metric_names)) # 按字母顺序排序metric名称

    # 2. 确定所有列的宽度
    # Stage Name 列的宽度
    stage_name_header = "Stage Name"
    stage_name_width = len(stage_name_header)
    for stage_name in data.keys():
        stage_name_width = max(stage_name_width, len(str(stage_name)))

    # Metric 列的宽度
    metric_widths = {}
    for metric_name in metric_names:
        metric_widths[metric_name] = len(metric_name) # 初始宽度为metric名称的长度
        for stage_data in data.values():
            # 考虑metric值，如果metric不存在则使用空字符串计算长度
            metric_value_str = str(stage_data.get(metric_name, ""))
            metric_widths[metric_name] = max(metric_widths[metric_name], len(metric_value_str))

    # 3. 打印表头
    header_line = f"{stage_name_header:<{stage_name_width}}" # 左对齐Stage Name列
    for metric_name in metric_names:
        # 右对齐Metric列，并在列之间添加一些间隔
        header_line += f"  {metric_name:>{metric_widths[metric_name]}}"
    print(header_line)

    # 4. 打印分隔线
    separator_line = "-" * stage_name_width
    for metric_name in metric_names:
        separator_line += "-" * (metric_widths[metric_name] + 2) # +2 是为了匹配间隔
    print(separator_line)

    # 5. 打印数据行
    for stage_name, stage_data in data.items():
        data_line = f"{str(stage_name):<{stage_name_width}}" # 左对齐Stage Name列
        for metric_name in metric_names:
            # 获取metric值，如果不存在则显示N/A或空字符串
            metric_value = stage_data.get(metric_name, "N/A")
            # 右对齐Metric值
            data_line += f"  {metric_value:>{metric_widths[metric_name]}.5g}"
        print(data_line)
