import os.path as path

def imgpath(basepath, klass, number):
    return path.join(basepath, "%s-%d.pgm" % (klass, number))
