### _PyShapelib 0.3.1 subversion_ ###
Customize 0.3 py shapelib to add support to (x,y,z) shape object.

(Orignial PyShapelib 0.3 is at: http://mail.python.org/pipermail/python-announce-list/2004-May/003129.html)

Source codes, Windows Installer, Mac Installer and Linux installer are available at:

**Download:** http://code.google.com/p/geoyard/downloads/list

```
import shapelib

# make shape file
shp_type = shapelib.SHPT_POLYGONZ

obj = shapelib.SHPObject(shp_type, 1, [[(0,0,0),(1,0,1),(1,1,2),(2,2,3),(3,3,4)]])
outfile = shapelib.create(shape_out, shp_type)
outfile.write_object( -1, obj)
del outfile

```


