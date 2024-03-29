# This file was automatically generated by SWIG (http://www.swig.org).
# Version 1.3.40
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.
# This file is compatible with both classic and new-style classes.

from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_shapelib', [dirname(__file__)])
        except ImportError:
            import _shapelib
            return _shapelib
        if fp is not None:
            try:
                _mod = imp.load_module('_shapelib', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _shapelib = swig_import_helper()
    del swig_import_helper
else:
    import _shapelib
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


class SHPObject(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SHPObject, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SHPObject, name)
    __repr__ = _swig_repr
    __swig_getmethods__["type"] = _shapelib.SHPObject_type_get
    if _newclass:type = _swig_property(_shapelib.SHPObject_type_get)
    __swig_getmethods__["id"] = _shapelib.SHPObject_id_get
    if _newclass:id = _swig_property(_shapelib.SHPObject_id_get)
    def __init__(self, *args): 
        this = _shapelib.new_SHPObject(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _shapelib.delete_SHPObject
    __del__ = lambda self : None;
    def extents(self): return _shapelib.SHPObject_extents(self)
    def vertices(self): return _shapelib.SHPObject_vertices(self)
SHPObject_swigregister = _shapelib.SHPObject_swigregister
SHPObject_swigregister(SHPObject)

class ShapeFile(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ShapeFile, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ShapeFile, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _shapelib.new_ShapeFile(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _shapelib.delete_ShapeFile
    __del__ = lambda self : None;
    def close(self): return _shapelib.ShapeFile_close(self)
    def info(self): return _shapelib.ShapeFile_info(self)
    def read_object(self, *args): return _shapelib.ShapeFile_read_object(self, *args)
    def write_object(self, *args): return _shapelib.ShapeFile_write_object(self, *args)
    def cobject(self): return _shapelib.ShapeFile_cobject(self)
ShapeFile_swigregister = _shapelib.ShapeFile_swigregister
ShapeFile_swigregister(ShapeFile)


def open(*args):
  return _shapelib.open(*args)
open = _shapelib.open

def create(*args):
  return _shapelib.create(*args)
create = _shapelib.create

def c_api():
  return _shapelib.c_api()
c_api = _shapelib.c_api

def type_name(*args):
  return _shapelib.type_name(*args)
type_name = _shapelib.type_name

def part_type_name(*args):
  return _shapelib.part_type_name(*args)
part_type_name = _shapelib.part_type_name
SHPT_NULL = _shapelib.SHPT_NULL
SHPT_POINT = _shapelib.SHPT_POINT
SHPT_ARC = _shapelib.SHPT_ARC
SHPT_POLYGON = _shapelib.SHPT_POLYGON
SHPT_MULTIPOINT = _shapelib.SHPT_MULTIPOINT
SHPT_POINTZ = _shapelib.SHPT_POINTZ
SHPT_ARCZ = _shapelib.SHPT_ARCZ
SHPT_POLYGONZ = _shapelib.SHPT_POLYGONZ
SHPT_MULTIPOINTZ = _shapelib.SHPT_MULTIPOINTZ
SHPT_POINTM = _shapelib.SHPT_POINTM
SHPT_ARCM = _shapelib.SHPT_ARCM
SHPT_POLYGONM = _shapelib.SHPT_POLYGONM
SHPT_MULTIPOINTM = _shapelib.SHPT_MULTIPOINTM
SHPT_MULTIPATCH = _shapelib.SHPT_MULTIPATCH
SHPP_TRISTRIP = _shapelib.SHPP_TRISTRIP
SHPP_TRIFAN = _shapelib.SHPP_TRIFAN
SHPP_OUTERRING = _shapelib.SHPP_OUTERRING
SHPP_INNERRING = _shapelib.SHPP_INNERRING
SHPP_FIRSTRING = _shapelib.SHPP_FIRSTRING
SHPP_RING = _shapelib.SHPP_RING


