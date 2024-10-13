# imports


# ---

class GraycartMaterial(object):

    def __init__(self, name, properties, thickness=0, etch_rates=None, dep_rates=None):
        super(GraycartMaterial, self).__init__()

        self.name = name
        self.properties = properties
        self.thickness = thickness
        self.etch_rates = etch_rates
        self.dep_rates = dep_rates

    def __repr__(self):
        class_ = 'GraycartMaterial'
        repr_dict = {'Material Name': self.name,
                     'Thickness': self.thickness,
                     }
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def apply_process(self, process, recipe, time):
        if process == 'Coat' and self.dep_rates is not None:
            self.thickness += self.dep_rates[recipe]
        elif process == 'Etch' and self.etch_rates is not None:
            self.thickness -= self.etch_rates[recipe] * time / 60


class SiliconMaterial(GraycartMaterial):
    def __init__(self, name, properties, thickness=0, etch_rates=None, dep_rates=None):
        if etch_rates is None:
            etch_rates = {
                'smOOth.V2': 0.0,
                'smOOth.V1': 0.0,
                'SF6+O2.V5': 2.25,  # um/min
                'SF6+O2.V6': 1.875,  # um/min (n = 3)
                'SF6+O2.V10': 2.36,  # (um/min); RECIPE: SF6/O2 flow: 55/50; Power: 800 W, Pressure: 10 mTorr
                'SF6+O2.V11': 2.52,  # (um/min); RECIPE: SF6/O2 flow: 60/50; Power: 800 W, Pressure: 10 mTorr
                'SF6+O2.V12': 2.64,  # (um/min); RECIPE: SF6/O2 flow: 65/50; Power: 800 W, Pressure: 10 mTorr
                'SF6+O2.V13': 2.7,  # (um/min); RECIPE: SF6/O2 flow: 70/50; Power: 800 W, Pressure: 10 mTorr
                'SF6+O2.V7': 3.00,  # um/min
                'SF6+O2.V8': 2.90,  # um/min
                'SF6': 5,  # um/min
            }
        GraycartMaterial.__init__(self, name, properties, thickness, etch_rates, dep_rates)


class PhotoresistMaterial(GraycartMaterial):
    def __init__(self, name, properties, thickness=0, etch_rates=None, dep_rates=None):
        if etch_rates is None:
            etch_rates = {
                'smOOth.V2': 0.550,  # um/min (n = 5)
                'smOOth.V1': 0.500,  # um/min
                'SF6+O2.V5': 0.25,  # um/min
                'SF6+O2.V6': 0.222,  # um/min (n = 9)
                'SF6+O2.V10': 0.21,  # (um/min); RECIPE: SF6/O2 flow: 55/50; Power: 800 W, Pressure: 10 mTorr
                'SF6+O2.V11': 0.2075,  # (um/min); RECIPE: SF6/O2 flow: 60/50; Power: 800 W, Pressure: 10 mTorr
                'SF6+O2.V12': 0.2025,  # (um/min); RECIPE: SF6/O2 flow: 65/50; Power: 800 W, Pressure: 10 mTorr
                'SF6+O2.V13': 0.20,  # (um/min); RECIPE: SF6/O2 flow: 70/50; Power: 800 W, Pressure: 10 mTorr
                'SF6+O2.V7': 0.05,  # um/min
                'SF6+O2.V8': 0.05,  # um/min
                'SF6': 0.01,  # um/min
            }
        if dep_rates is None:
            """ 
            NOTE: 'dep_rates' refers to the spinner recipe, as outlined below:
            
            Recipe  Spin Speed (rpm)    Spin Time (s)       Thickness (um)
            4       2500                30                  8.125
            5       3000                30                  7.45
            6       3500                30                  6.95
            7       4000                30                  6.45
            """
            dep_rates = {
                4: 8.125,
                5: 7.45,
                6: 6.95,
                7: 6.45,
            }
        GraycartMaterial.__init__(self, name, properties, thickness, etch_rates, dep_rates)