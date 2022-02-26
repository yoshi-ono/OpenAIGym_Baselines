import logging

def Init():
    format = "%(asctime)s %(levelname)s %(module)s(%(lineno)s):%(funcName)s\t%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=format)

def GetLogger():
    return logging.getLogger()
