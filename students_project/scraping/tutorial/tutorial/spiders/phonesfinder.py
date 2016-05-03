# -*- coding: utf-8 -*-
import scrapy

from tutorial.items import TutorialItem


class PhonesfinderSpider(scrapy.Spider):
    name = "phonesfinder"
    allowed_domains = ["www.phonearena.com"]
    start_urls = [
        'http://www.phonearena.com/phones#/phones/page/1',
    ]

    def parse(self, response):
        for i in range(1, 203, 1):
            url = "http://www.phonearena.com/phones/page/%d" % i
            print "Request: " + url
            yield scrapy.Request(url, self.parse_page)
        pass

    def parse_page(self, response):
        for alink in response.xpath('//a[@class="atext"]'):
            link = alink.xpath('@href').extract()[0]
            #print alink.xpath('@href').extract()
            item = TutorialItem()
            item['phoneLink'] = link
            yield item
        return