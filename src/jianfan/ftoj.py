# -*- coding: utf-8 -*-
# author: Leon Dong <Leon.Dong@gmail.com>
# commiter: Thomas <tsroten@gmail.com>

"""
    Jianfan is a library for translation between traditional and simplified chinese.
    Support Python 2 and Python 3. Thanks for Thomas to provide Python 3 support.
        two functions are provided by the library:
        jtof: translate simplified chinese to traditional chinese
        jtoj: translate traditional chinese to simplified chinese
        the two functions accept one parameter that is unicode or string
        the type of return value is unicode

    Jianfan是一个简体中文和繁体中文转换的库。提供了两个函数：
        jtof: 简体转换为繁体
        ftoj: 繁体转换为简体
        函数接受unicode和string类型作为参数，返回值统一为unicode
"""

import codecs
import os
import sys
from __init__ import _t
from charsets import gbk, big5

'''
def ftoj(unicode_string):
    """
        Translate traditional chinese to simplified chinese.
        >>> t = u'中華'
        >>> print ftoj(t)
        中华
    """
    return _t(unicode_string, big5, gbk)
'''

def main():
    input_file_path = sys.argv[1];
    output_file_path = sys.argv[2];
    
    input_stream = codecs.open(input_file_path, 'r', 'utf-8');
    output_stream = codecs.open(output_file_path, 'w', 'utf-8');
    
    for line in input_stream:
        line = line.strip();
        output_stream.write("%s\n" % (_t(line, big5, gbk)));

if __name__ == '__main__':
    main()