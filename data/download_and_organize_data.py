"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2024-09-26 18:48:02
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2024-09-26 18:50:47
FilePath: /root/osi-450-a/data/download_and_organize_data.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""

import os
from datetime import datetime, timedelta
import requests
import shutil
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def download_file(file_url, output_file, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = requests.get(file_url)
            response.raise_for_status()  # Raise HTTPError for bad responses

            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {output_file}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed to download {file_url}: {e}")
            time.sleep(delay)

    print(f"Failed to download {file_url} after {retries} attempts")
    return False


def organize_file(current_directory, filename):
    # 匹配两种文件名格式
    pattern = (
        r"^ice_conc_nh_ease2-250_(cdr-v3p0|icdr-v3p0)_(\d{4})(\d{2})\d{2}1200\.nc$"
    )
    match = re.match(pattern, filename)

    if match:
        year, month = match.groups()[1:]
        year_directory = os.path.join(current_directory, year)
        month_directory = os.path.join(year_directory, month)

        os.makedirs(month_directory, exist_ok=True)

        source_path = os.path.join(current_directory, filename)
        target_path = os.path.join(month_directory, filename)

        shutil.move(source_path, target_path)


def download_and_organize_data(start_date, end_date, output_directory, max_workers=5):
    base_url_pre_2021 = "https://thredds.met.no/thredds/fileServer/osisaf/met.no/reprocessed/ice/conc_450a_files"
    base_url_post_2021 = "https://thredds.met.no/thredds/fileServer/osisaf/met.no/reprocessed/ice/conc_cra_files"
    tasks = []

    current_date = start_date
    while current_date <= end_date:
        file_date = current_date.strftime("%Y%m%d")

        if current_date.year < 2021:
            base_url = base_url_pre_2021
            filename = f"ice_conc_nh_ease2-250_cdr-v3p0_{file_date}1200.nc"
        else:
            base_url = base_url_post_2021
            filename = f"ice_conc_nh_ease2-250_icdr-v3p0_{file_date}1200.nc"

        file_url = f"{base_url}/{current_date.year}/{current_date.month:02d}/{filename}"
        output_file = os.path.join(output_directory, filename)

        tasks.append((file_url, output_file, filename))

        current_date += timedelta(days=1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(download_file, file_url, output_file): (
                file_url,
                output_file,
                filename,
            )
            for file_url, output_file, filename in tasks
        }

        for future in as_completed(future_to_task):
            file_url, output_file, filename = future_to_task[future]
            try:
                success = future.result()
                if success:
                    organize_file(output_directory, filename)
            except Exception as e:
                print(f"Failed to process {file_url}: {e}")


if __name__ == "__main__":
    start_date = datetime(1991, 1, 1)
    end_date = datetime(2020, 12, 31)
    output_directory = "/data2/OSI-SAF"

    os.makedirs(output_directory, exist_ok=True)

    download_and_organize_data(start_date, end_date, output_directory)
