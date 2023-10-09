import scrapy
import json
import re
import csv

class WeatherSpider(scrapy.Spider):
    name = 'weather_spider'
    allowed_domains = ['tianqi.2345.com']

    # Start requests method
    def start_requests(self):
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
                    yield scrapy.Request(url, self.parse, meta={'city_name': city_name})

    # Parse method
    def parse(self, response):
        html_data = json.loads(response.text)['data']
        tr_pattern = r'<tr>(.*?)</tr>'
        td_pattern = r'<td>(.*?)</td>'
        trs = re.findall(tr_pattern, html_data, re.DOTALL | re.IGNORECASE)[1:]  # [1:] to skip the first header row
        for tr in trs:
            tds = re.findall(td_pattern, tr, re.DOTALL | re.IGNORECASE)
            tds.append(response.meta['city_name'])
            yield {
                '日期': tds[0],
                '最高温度': tds[1],
                '最低温度': tds[2],
                '天气': tds[3],
                '风向': tds[4],
                '城市': tds[5]
            }
