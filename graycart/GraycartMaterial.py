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
        """
        old_etch_rates = {          NOTE: the DSEiii chamber temperature has been changed, so rates likely different
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
        """
        if etch_rates is None:
            etch_rates = {
                'smOOth.V1': 0.0,
                'SF6+O2.V1': 2.0,  # (um/min); DAMIEN'S RECIPE: SF6/O2 flow: ?/?
                'SF6+O2.V2': 3.2,  # (um/min); RECIPE: SF6/O2 flow: 80/50
                'SF6+O2.V4': 3.55,  # (um/min); RECIPE: SF6/O2 flow: 90/50
                'SF6+O2.V6': 4.0,  # (um/min); RECIPE: SF6/O2 flow: 110/50
                'SF6+O2.V8': 4.27,  # (um/min); RECIPE: SF6/O2 flow: 130/50
                'SF6+O2.V48': 5.2,  # (um/min); RECIPE: SF6/O2 flow: 130/40
                'SF6+O2.S25': 2.5,  # SELECTIVITY 25
            }
        GraycartMaterial.__init__(self, name, properties, thickness, etch_rates, dep_rates)


class PhotoresistMaterial(GraycartMaterial):
    def __init__(self, name, properties, thickness=0, etch_rates=None, dep_rates=None):
        """
        old_etch_rates = {          NOTE: the DSEiii chamber temperature has been changed, so rates likely different
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
        """
        if etch_rates is None:
            etch_rates = {
                'smOOth.V1': 0.750,  # um/min (n = 1)
                'SF6+O2.V1': 0.24,  # (um/min); DAMIEN'S RECIPE: SF6/O2 flow: ?/?
                'SF6+O2.V2': 0.27,  # (um/min); RECIPE: SF6/O2 flow: 80/50
                'SF6+O2.V4': 0.225,  # (um/min); RECIPE: SF6/O2 flow: 90/50
                'SF6+O2.V6': 0.175,  # (um/min); RECIPE: SF6/O2 flow: 110/50
                'SF6+O2.V8': 0.145,  # (um/min); RECIPE: SF6/O2 flow: 130/50        (need more data to verify)
                'SF6+O2.V48': 0.16,  # (um/min); RECIPE: SF6/O2 flow: 130/40       (need more data; don't trust this)
                'SF6+O2.S25': 0.1,  # SELECTIVITY 25
            }
        if dep_rates is None:
            """ 
            NOTE: 'dep_rates' refers to the spinner recipe, as outlined below:
            
            Recipe  Spin Speed (rpm)    Spin Time (s)       Thickness (um)
            4       2500                30                  8.125
            5       3000                30                  7.45                (recently, 12/1/24: ~7.2 um)
            6       3500                30                  6.95
            7       4000                30                  6.45
            """
            dep_rates = {
                4: 8.125,
                5: 7.45,
                5.5: 7.2,  # This is a special case, when recipe 5 was used but actual thickness is less than
                6: 6.95,
                7: 6.45,
            }
        GraycartMaterial.__init__(self, name, properties, thickness, etch_rates, dep_rates)