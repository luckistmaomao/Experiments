#coding:utf-8

__author__ = 'yuzt'

def strdecode(s):
    if not isinstance(s, unicode):
        try:
            s = s.decode('utf-8')
        except UnicodeError:
            s = s.decode('gbk', 'ignore')
    return s
