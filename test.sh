#!/bin/bash
###
 # @Author: 爱吃菠萝 1690608011@qq.com
 # @Date: 2024-06-25 16:06:28
 # @LastEditors: 爱吃菠萝 1690608011@qq.com
 # @LastEditTime: 2024-09-27 15:32:47
 # @FilePath: /root/OSI-SAF/test.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
### 

START_DATE="20160101"
END_DATE="20201231"

INPUT_LENGTH=14
PREDICTION_LENGTH=14

DAYS_TO_ADD=$((INPUT_LENGTH + PREDICTION_LENGTH - 1))

# 将日期字符串转换为自纪元以来的秒数
start_seconds=$(date -d "$START_DATE" +%s)
end_seconds=$(date -d "$END_DATE" +%s)

# 计算调整后的结束日期秒数，减去 DAYS_TO_ADD
adjusted_end_seconds=$(date -d "$END_DATE - $DAYS_TO_ADD days" +%s)

# 在范围内循环遍历每一天
current_seconds=$start_seconds
while [ $current_seconds -le $adjusted_end_seconds ]; do
    current_start_date=$(date -d "@$current_seconds" +%Y%m%d)
    current_end_date=$(date -d "$current_start_date + $DAYS_TO_ADD days" +%Y%m%d)
    
    # 执行测试命令
    python test.py -st $current_start_date -et $current_end_date
    
    # 检查上一条命令的退出状态
    if [ $? -ne 0 ]; then
        echo "Error occurred while executing python test.py -st $current_start_date -et $current_end_date"
        exit 1
    fi

    # 移动到下一天
    current_seconds=$(date -d "$current_start_date + 1 day" +%s)
done