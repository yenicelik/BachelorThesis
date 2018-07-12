import numpy as np

X_sinusoidal = np.asarray([
    [0., 0., 0., 0., 0.],
    [-2.31425797, -4.33765134, -2.55495294, -0.46821858, 3.9693806],
    [-0.40695843, -0.72294189, -4.81238808, -0.82142196, 0.80603318],
    [-0.40695843, -0.72294189, -0.80206468, -0.13690366, 4.83619908],
    [-2.44175058, -4.33765134, -4.81238808, -0.82142196, 0.80603318],
    [-0.40695843, -4.33765134, -0.80206468, -0.13690366, 0.80603318],
    [-2.44175058, -2.4869915, -0.80206468, -0.13690366, 0.80603318],
    [-0.40695843, -0.72294189, -4.81238808, -0.82142196, 4.83619908],
    [-0.40695843, -4.33765134, -4.81238808, -0.82142196, 4.83619908],
    [-0.40695843, -4.33765134, -3.31417012, -0.13690366, 0.80603318],
    [-2.44175058, -0.72294189, -3.55897258, -0.13690366, 2.72591677],
    [-0.40695843, -4.33765134, -0.80206468, -0.82142196, 4.83619908],
    [-0.40695843, -2.41371836, -0.80206468, -0.82142196, 2.44488807],
    [-0.40695843, -2.73368031, -4.81238808, -0.13690366, 2.74200906],
    [-2.44175058, -2.31337942, -0.80206468, -0.82142196, 4.83619908],
    [-0.40695843, -0.72294189, -2.28612262, -0.82142196, 0.80603318],
    [-2.44175058, -1.8252745, -4.81238808, -0.13690366, 0.80603318],
    [-2.44175058, -4.33765134, -1.8575642, -0.82142196, 0.80603318],
    [-2.44175058, -2.58613706, -4.81238808, -0.82142196, 4.83619908],
    [-2.44175058, -0.72294189, -0.80206468, -0.13690366, 2.83589407],
    [-0.40695843, -2.62402282, -2.80747033, -0.13690366, 4.83619908],
    [-2.44175058, -4.33765134, -4.81238808, -0.13690366, 3.1750571],
    [-0.40695843, -4.33765134, -2.29669937, -0.82142196, 2.87932756],
    [-2.44175058, -0.72294189, -2.73816306, -0.82142196, 4.83619908],
    [-2.44175058, -2.74422506, -2.98776398, -0.82142196, 2.27673562],
    [-2.44175058, -0.72294189, -4.81238808, -0.82142196, 4.15190968],
    [-0.40695843, -2.31249116, -3.57947243, -0.13690366, 0.80603318],
    [-0.40695843, -0.72294189, -0.80206468, -0.13690366, 2.42828011],
    [-2.44175058, -4.33765134, -0.80206468, -0.13690366, 2.90880007],
    [-2.44175058, -0.72294189, -2.46127595, -0.82142196, 0.80603318],
    [-0.40695843, -2.38411131, -0.80206468, -0.13690366, 0.80603318],
    [-0.40695843, -0.72294189, -3.23565267, -0.13690366, 3.272051],
    [-0.40695843, -4.33765134, -4.81238808, -0.82142196, 1.9796675],
    [-2.44175058, -4.33765134, -4.81238808, -0.82142196, 4.83619908],
    [-2.44175058, -2.6892151, -2.77520995, -0.13690366, 4.83619908],
    [-0.40695843, -4.33765134, -2.87213413, -0.13690366, 4.83619908],
    [-2.44175058, -0.72294189, -0.80206468, -0.13690366, 0.80603318],
    [-2.44175058, -2.56731363, -0.80206468, -0.13690366, 3.08329903],
    [-0.40695843, -0.72294189, -4.81238808, -0.82142196, 2.71729436],
    [-0.87409685, -0.72294189, -0.80206468, -0.13690366, 0.80603318],
    [-0.40695843, -2.46452432, -2.90582395, -0.82142196, 2.81597073],
    [-0.40695843, -4.33765134, -0.80206468, -0.13690366, 2.89306648],
    [-2.44175058, -4.33765134, -0.80206468, -0.13690366, 4.83619908],
    [-0.40695843, -2.48064358, -0.80206468, -0.82142196, 4.83619908],
    [-0.40695843, -3.24469102, -2.06693866, -0.82142196, 0.80603318],
    [-0.40695843, -0.72294189, -2.62786747, -0.82142196, 4.83619908],
    [-2.44175058, -4.33765134, -3.41263392, -0.82142196, 2.14053407],
    [-2.44175058, -2.37231491, -4.81238808, -0.82142196, 2.61450039],
    [-0.89700819, -3.312308, -4.81238808, -0.13690366, 0.80603318],
    [-0.40695843, -2.52683898, -4.81238808, -0.82142196, 4.83619908],
    [-1.44222878, -3.22997473, -1.97571277, -0.13690366, 2.11289721],
    [-0.40695843, -4.33765134, -3.98883949, -0.13690366, 3.31438283],
    [-2.44175058, -2.91113586, -3.19975318, -0.13690366, 0.80603318],
    [-1.49231656, -3.15477155, -3.68292954, -0.82142196, 3.81697104],
    [-1.45223735, -1.73114181, -1.88304724, -0.13690366, 3.70726661],
    [-2.44175058, -0.72294189, -4.81238808, -0.82142196, 0.80603318],
    [-1.4637488, -3.46670877, -1.86760696, -0.82142196, 4.83619908],
    [-2.44175058, -1.87065597, -2.22653342, -0.82142196, 3.57791129],
    [-2.44175058, -0.72294189, -0.80206468, -0.82142196, 4.83619908],
    [-1.44145186, -0.72294189, -2.17399285, -0.82142196, 2.4408575],
    [-1.55008597, -4.33765134, -0.80206468, -0.82142196, 1.62981347],
    [-2.44175058, -1.85532463, -0.80206468, -0.82142196, 1.95632329],
    [-0.40695843, -2.97547448, -0.80206468, -0.13690366, 3.77168527],
    [-1.38990137, -0.72294189, -3.65052773, -0.13690366, 0.80603318],
    [-1.59993188, -1.39392336, -4.81238808, -0.13690366, 4.83619908],
    [-2.44175058, -4.33765134, -3.35907988, -0.82142196, 4.83619908],
    [-1.7801692, -0.72294189, -4.81238808, -0.13690366, 2.21882272],
    [-1.37328884, -1.48508476, -3.77383998, -0.82142196, 3.79251001],
    [-0.40695843, -0.72294189, -0.80206468, -0.82142196, 3.6424574],
    [-1.63911115, -3.35106705, -4.81238808, -0.82142196, 1.87362726],
    [-0.40695843, -2.20510561, -4.81238808, -0.82142196, 1.28376033],
    [-0.40695843, -4.33765134, -2.03909444, -0.13690366, 1.66055672],
    [-1.68014516, -2.1416339, -2.15339035, -0.82142196, 0.80603318],
    [-0.40695843, -1.7679093, -2.23679639, -0.13690366, 1.89100959],
    [-1.45933195, -1.73871034, -3.72365574, -0.82142196, 1.7769061],
    [-1.67827324, -3.5587342, -0.80206468, -0.82142196, 3.624296],
    [-0.40695843, -0.72294189, -3.49903058, -0.82142196, 1.87244724],
    [-1.56319538, -0.72294189, -1.75008375, -0.82142196, 4.03883697],
    [-0.40695843, -2.18132732, -1.95727142, -0.82142196, 3.89510586],
    [-1.65615056, -4.33765134, -3.42940771, -0.82142196, 0.80603318],
    [-1.46015811, -1.94950244, -0.80206468, -0.13690366, 4.83619908],
    [-2.44175058, -2.35968004, -3.9028127, -0.13690366, 3.57115396],
    [-0.40695843, -0.72294189, -4.81238808, -0.13690366, 3.81484959],
    [-1.31227227, -4.33765134, -3.71648163, -0.13690366, 2.16076627],
    [-1.40720137, -3.53383273, -4.81238808, -0.13690366, 4.83619908],
    [-2.44175058, -2.87752039, -4.81238808, -0.82142196, 0.80603318],
    [-2.44175058, -3.71412467, -1.86160269, -0.82142196, 2.73550452],
    [-0.92765912, -3.44145947, -2.35098475, -0.13690366, 3.58153545],
    [-0.40695843, -3.36875785, -0.80206468, -0.13690366, 1.79333632],
    [-0.40695843, -1.43231367, -3.76356161, -0.13690366, 4.83619908],
    [-2.44175058, -1.65572782, -2.21554579, -0.13690366, 1.91369913],
    [-2.44175058, -0.72294189, -3.89550831, -0.13690366, 4.83619908],
    [-0.40695843, -4.33765134, -4.81238808, -0.13690366, 0.80603318],
    [-1.49369963, -1.96627742, -2.8745249, -0.82142196, 4.83619908],
    [-0.40695843, -3.24313188, -3.48564757, -0.13690366, 2.03311874],
    [-2.44175058, -4.33765134, -2.30168059, -0.13690366, 1.88363124],
    [-0.40695843, -3.47857103, -3.58586777, -0.82142196, 4.83619908],
    [-0.40695843, -2.40016577, -3.81680013, -0.13690366, 3.6990514],
    [-2.44175058, -3.34164638, -0.80206468, -0.82142196, 1.6287604],
    [-2.44175058, -3.3851912, -4.30058718, -0.13690366, 1.86834153]
])

Y_sinusoidal = np.asarray([
    [0.00000000e+00],
    [-3.96101751e-02],
    [1.76625015e-01],
    [-4.62138426e-01],
    [7.57395973e-02],
    [9.90804013e-01],
    [-2.06571693e-01],
    [-2.30434301e-02],
    [6.00905785e-02],
    [-5.39168134e-01],
    [-6.30453337e-01],
    [1.80113795e-01],
    [-2.35249414e-01],
    [-2.08495587e-01],
    [2.28929498e-01],
    [-8.34799662e-01],
    [2.04963269e-01],
    [-6.10294731e-01],
    [-1.25951903e-01],
    [-3.58580539e-01],
    [-3.00235612e-02],
    [-6.13031270e-02],
    [2.57191593e-02],
    [-2.61252796e-01],
    [1.01666406e-01],
    [1.22813708e-01],
    [-2.69070304e-02],
    [-4.44444901e-01],
    [-5.39574996e-02],
    [-2.72980840e-01],
    [3.46624509e-01],
    [-3.31982467e-01],
    [5.70041388e-01],
    [8.24197912e-02],
    [-2.98318863e-02],
    [4.26317113e-03],
    [-5.66817113e-01],
    [4.72685341e-01],
    [-1.45489810e-01],
    [-7.36847551e-01],
    [4.13665496e-01],
    [-3.75759506e-01],
    [-2.23132223e-01],
    [2.95063271e-01],
    [-3.48248501e-01],
    [-4.37984148e-01],
    [-3.54773407e-02],
    [-8.35334867e-02],
    [1.25292126e-01],
    [-2.34507130e-02],
    [7.34932879e-02],
    [9.73121846e-02],
    [4.64867792e-02],
    [-3.34662103e-01],
    [6.29749957e-02],
    [1.47789355e-01],
    [-2.69572431e-01],
    [2.61885035e-01],
    [-6.60598949e-01],
    [-4.17011800e-01],
    [-7.27590268e-01],
    [-3.31333620e-02],
    [1.92796089e-01],
    [-2.64660053e-01],
    [1.78959906e-01],
    [-2.95672279e-02],
    [-2.36090538e-01],
    [-9.01167015e-02],
    [-2.63744338e-01],
    [3.55993145e-02],
    [2.25845029e-01],
    [-4.06907780e-01],
    [2.65381131e-03],
    [-2.31103956e-01],
    [3.78745299e-01],
    [1.11925163e-01],
    [-3.74239117e-02],
    [-6.19079958e-01],
    [4.49135318e-01],
    [-3.03238693e-01],
    [1.41431173e-01],
    [-8.18676413e-03],
    [-2.54903408e-02],
    [7.25538109e-02],
    [-9.89744680e-02],
    [8.66253607e-02],
    [-1.20672963e-01],
    [1.65843243e-01],
    [-2.72408048e-01],
    [1.54544855e-01],
    [2.17255394e-01],
    [1.25872377e-01],
    [2.79385803e-02],
    [8.19187000e-02],
    [1.21743639e-01],
    [7.18941028e-02],
    [-1.38974439e-01],
    [-9.03674873e-02],
    [-5.04591821e-01],
    [9.36043756e-04]
])

def get_sinusoidal_training():
    assert X_sinusoidal.shape[0] == Y_sinusoidal.shape[0], ("Data does not conform shapes!")
    return X_sinusoidal, Y_sinusoidal

