import json

from scrapy import *
from urllib.parse import urljoin
import logging

logging.getLogger("scrapy.spidermiddlewares.depth").setLevel(logging.INFO)

class PdfSpider(Spider):
    name = "pdfspider"

    def start_requests(self):
        urls = [
            # 'http://h2t.anthropomatik.kit.edu/101.php',
            # 'http://h2t.anthropomatik.kit.edu/index.php'
            # 'https://www.informatik.kit.edu/226.php'
            'https://www.stanford.edu/list/academic/',
            'https://cs.stanford.edu/'
        ]
        for url in urls:
            yield Request(url=url, callback=self.parse)

    def parse(self, response):
        if response.headers["content-type"].decode("utf-8") == 'application/pdf':
            self.save_pdf(response)
        else:
            for url in response.css('a[href]::attr(href)').extract():
                url = str(urljoin(response.url, url))
                if not url.startswith("javascript") \
                        and not url.startswith("mailto") \
                        and ".stanford.edu" in url:
                    yield Request(url, callback=self.parse)

    def save_pdf(self, response):
        path = self.get_path(response.url)
        with open(path, "wb") as f:
            f.write(response.body)
        with open(path + ".json", "w") as f:
            metadata = {
                'url': response.url,
                'parent_url': response.request.headers['referer'].decode("utf-8")
            }
            f.write(json.dumps(metadata))

    def get_path(self, url):
        return "output/" + "".join([c for c in url if c.isalpha() or c.isdigit() or c in [' ', '.']]).rstrip()
