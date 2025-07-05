import enginesim as es
import matplotlib.pyplot as plt

# es.OFsweep(
#     OFstart = 0.5,
#     OFend = 10,
#     fuel = 'Ethanol',
#     ox = 'N2O',
#     pc = 30,
#     pe = 1.3,
#     cr = 8,
# )

es.OFsweep(
    OFstart = 0.5,
    OFend = 10,
    fuel = 'Isopropanol',
    ox = 'N2O',
    pc = 30,
    pe = 1.3,
    cr = 8,
)
plt.savefig('common_figs/OF_sweep_isopropanol.png', dpi=300, bbox_inches='tight')
# plt.show(block=True)