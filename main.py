import numpy as np
from pint import UnitRegistry, Quantity
ureg = UnitRegistry()
Q_ = ureg.Quantity

Pc = 1000
Tc = 6140
cstar = 5660
Dt = 24.9
R = 11.71
Cp = 0.485
pr = 0.816
visc = 4.18e-6
g = 32.2
gamma = 1.2
M = 1
TwgTc = 0.8

correction_factor = 1 / ((0.5 * TwgTc * (1 + 0.5 * (gamma - 1) * M**2)+0.5)**0.68 * (1 + 0.5 * (gamma - 1) * M**2)**0.12)

hg = Q_((0.026 / Dt**0.2) * (visc**0.2 * Cp / pr**0.6) * (Pc * g / cstar)**0.8 * (Dt / R)**0.1, "Btu / (inch ** 2 * second)").to("W / (m**2)")
print(hg)



([[ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 262.47088751,  262.47088751],
       [ 277.73067765,  277.73067765],
       [ 300.81258027,  300.81258027],
       [ 327.02206881,  327.02206881],
       [ 356.96328195,  356.96328195],
       [ 391.39556893,  391.39556893],
       [ 431.28435289,  431.28435289],
       [ 477.87267936,  477.87267936],
       [ 532.78369233,  532.78369233],
       [ 598.1703707 ,  598.1703707 ],
       [ 676.93924838,  676.93924838],
       [ 773.09310426,  773.09310426],
       [ 892.27074875,  892.27074875],
       [1035.45107517, 1035.45107517],
       [1180.27032113, 1180.27032113],
       [1325.55045103, 1325.55045103],
       [1471.41591984, 1471.41591984],
       [1616.90798998, 1616.90798998],
       [1760.51241504, 1760.51241504],
       [1900.18209794, 1900.18209794],
       [2033.43478604, 2033.43478604],
       [2157.58640009, 2157.58640009],
       [2269.76668881, 2269.76668881],
       [2367.0142639 , 2367.0142639 ],
       [2446.65699238, 2446.65699238],
       [2506.44082935, 2506.44082935],
       [2544.2300044 , 2544.2300044 ],
       [2556.82805489, 2556.82805489],
       [2531.65186261, 2531.65186261],
       [2413.98095319, 2413.98095319],
       [2223.70115596, 2223.70115596],
       [2051.2800628 , 2051.2800628 ],
       [1902.45761395, 1902.45761395],
       [1772.6952747 , 1772.6952747 ],
       [1658.69306618, 1658.69306618],
       [1557.90810475, 1557.90810475],
       [1468.32088598, 1468.32088598],
       [1388.30066695, 1388.30066695],
       [1316.51423572, 1316.51423572],
       [1251.86063119, 1251.86063119],
       [1193.42286643, 1193.42286643],
       [1140.43141499, 1140.43141499],
       [1092.23612676, 1092.23612676],
       [1048.28433595, 1048.28433595],
       [1008.1036035 , 1008.1036035 ],
       [ 971.28798011,  971.28798011],
       [ 937.48697726,  937.48697726],
       [ 906.39664334,  906.39664334],
       [ 877.75229202,  877.75229202],
       [ 851.32253813,  851.32253813],
       [ 826.90437634,  826.90437634],
       [ 804.31909739,  804.31909739],
       [ 783.40888137,  783.40888137],
       [ 764.03394172,  764.03394172],
       [ 746.07716152,  746.07716152],
       [ 729.42541629,  729.42541629],
       [ 713.97541816,  713.97541816],
       [ 699.63884512,  699.63884512],
       [ 686.33650212,  686.33650212],
       [ 673.99955974,  673.99955974],
       [ 662.57249181,  662.57249181],
       [ 651.98643384,  651.98643384],
       [ 642.1892261 ,  642.1892261 ],
       [ 633.13368067,  633.13368067]])