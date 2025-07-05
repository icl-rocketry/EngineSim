import adsk.core, adsk.fusion, adsk.cam, traceback
import csv
import math

def run(context):
    ui = None
    try:
    
        app = adsk.core.Application.get()
        ui  = app.userInterface
        ui.messageBox('ContourTest')
        design = app.activeProduct
        rootComp = design.rootComponent 

        
        ############# preprocessing ####################
        y = []
        z = []

        with open('C:\\Users\\Martin\\OfflineOnly\\ICLR\\Cooling-Channel-Contour-Gen\\contourtest.csv', 'r') as file:
            my_reader = csv.reader(file, delimiter=',')
            for row in my_reader:
                y.append(float(row[0]))
                z.append(float(row[1]))



        ###### adding splines ############
        sketches = rootComp.sketches
        sketch = sketches.add(rootComp.xZConstructionPlane)
        lines = sketch.sketchCurves.sketchLines
        
        
        points = adsk.core.ObjectCollection.create()
        p = adsk.core.Point3D.create(0, z[0], y[0])
        for i in range(1, len(z)):
            p2 = adsk.core.Point3D.create(0, z[i], y[i])
            lines.addByTwoPoints(p, p2)
            p = p2

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))