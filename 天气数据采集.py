
import requests     # 发送请求要用的模块 需要额外安装的
import parsel
import csv
import scrapy


f = open('天气-gbk.csv', mode='a', encoding='gbk', newline='')
csv_writer = csv.writer(f)
#写入标题行
csv_writer.writerow(['日期', '最高温度', '最低温度', '天气', '风向', '城市'])

city_list = [54511, 58362, 59287, 59493]
for city in city_list:
    city_name = ''
    if city == 54511:
        city_name = '北京'
    elif city == 58362:
        city_name = '上海'
    elif city == 59287:
        city_name = '广州'
    elif city == 59493:
        city_name = '深圳'
    for year in range(2013, 2023):
        for month in range(1, 13):
            url = f'https://tianqi.2345.com/Pc/GetHistory?areaInfo%5BareaId%5D={city}&areaInfo%5BareaType%5D=2&date%5Byear%5D={year}&date%5Bmonth%5D={month}'
            # 1. 发送请求
            response = requests.get(url=url)
            # 2. 获取数据
            #表示从JSON字典中获取键为'data'的值。假设返回的JSON数据中包含一个键为'data'的字段，这行代码将提取出该字段的值并将其存储在html_data变量中。
            html_data = response.json()['data']
            # 3. 解析数据
            select = parsel.Selector(html_data)
            trs = select.css('.history-table tr')   # 拿到31个tr
            for tr in trs[1:]:                      # 第一个表头不要
                tds = tr.css('td::text').getall()   # 针对每个tr进行提取 取出所有的td里面的内容
                tds.append(city_name)               # 把城市追加到列表里面
                print(tds)
                # 4. 保存数据
                csv_writer.writerow(tds)

