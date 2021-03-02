import numpy as np


class RangeDict(dict):
    def __getitem__(self, item):
        if type(item) != range:  # change to xrange in Python 2
            for key in self:
                if item in key:
                    return self[key]
            raise KeyError(item)
        else:
            return super().__getitem__(item)


values = {}

########################
###### Hemoglobin ######
########################
values["Hemoglobin"] = {}
values["Hemoglobin"]["M"] = RangeDict(
    {range(2): [10.4, 12.6], range(2, 12): [11.0, 13.4], range(12, 150): [13.4, 17.0],}
)
values["Hemoglobin"]["F"] = RangeDict(
    {range(2): [10.4, 12.6], range(2, 12): [11.0, 13.4], range(12, 150): [11.7, 15.3],}
)

########################
###### Leukocytter #####
########################
values["Leukocytter"] = {}
values["Leukocytter"]["M"] = RangeDict(
    {range(2): [6.3, 15.5], range(2, 18): [4.4, 12.5], range(18, 150): [4.0, 11.0],}
)
values["Leukocytter"]["F"] = values["Leukocytter"]["M"]

########################
####### Natrium ########
########################
values["Natrium"] = {}
values["Natrium"]["M"] = RangeDict({range(18): [135, 146], range(18, 150): [137, 145],})
values["Natrium"]["F"] = values["Natrium"]["M"]

########################
######### CRP ##########
########################
values["CRP"] = {}
values["CRP"]["M"] = RangeDict({range(150): [0, 4]})
values["CRP"]["F"] = values["CRP"]["M"]

########################
######## Kalium ########
########################
values["Kalium"] = {}
values["Kalium"]["M"] = RangeDict({range(18): [3.7, 4.8], range(18, 150): [3.6, 4.6],})
values["Kalium"]["F"] = values["Kalium"]["M"]

########################
####### Albumin ########
########################
values["Albumin"] = {}
values["Albumin"]["M"] = RangeDict({range(18): [39, 50], range(18, 150): [39.7, 49.4],})
values["Albumin"]["F"] = values["Albumin"]["M"]

########################
####### Kreatinin ######
########################
values["Kreatinin"] = {}
values["Kreatinin"]["M"] = RangeDict(
    {
        range(11): [28, 57],
        range(11, 13): [37, 63],
        range(13, 15): [40, 72],
        range(15, 150): [60, 105],
    }
)
values["Kreatinin"]["F"] = RangeDict(
    {
        range(11): [28, 57],
        range(11, 13): [37, 63],
        range(13, 15): [40, 72],
        range(15, 150): [45, 90],
    }
)

########################
######### Trombocytter #
########################
values["Trombocytter"] = {}
values["Trombocytter"]["M"] = RangeDict(
    {range(18): [150, 460], range(18, 150): [150, 450],}
)
values["Trombocytter"]["F"] = values["Trombocytter"]["M"]

########################
####### ALAT ###########
########################
values["ALAT"] = {}
values["ALAT"]["M"] = RangeDict(
    {
        range(9): [8, 28],
        range(9, 14): [8, 37],
        range(14, 18): [8, 47],
        range(18, 150): [10, 70],
    }
)
values["ALAT"]["F"] = RangeDict(
    {range(9): [8, 35], range(9, 18): [9, 32], range(18, 150): [10, 45],}
)

########################
##### Bilirubin total ##
########################
values["Bilirubin total"] = {}
values["Bilirubin total"]["M"] = RangeDict(
    {range(18): [0, 25], range(18, 150): [5, 25],}
)
values["Bilirubin total"]["F"] = values["Bilirubin total"]["M"]

########################
####### ASAT ###########
########################
values["ASAT"] = {}
values["ASAT"]["M"] = RangeDict({range(18): [10, 45], range(18, 150): [15, 45],})
values["ASAT"]["F"] = RangeDict({range(18): [10, 45], range(18, 150): [15, 35],})

########################
####### ALP ############
########################
values["ALP"] = {}
values["ALP"]["M"] = RangeDict(
    {
        range(5): [100, 400],
        range(5, 9): [130, 370],
        range(9, 14): [110, 460],
        range(14, 17): [55, 410],
        range(17, 18): [55, 240],
        range(18, 150): [35, 105],
    }
)
values["ALP"]["F"] = RangeDict(
    {
        range(5): [100, 400],
        range(5, 13): [140, 400],
        range(13, 17): [40, 290],
        range(17, 18): [40, 105],
        range(18, 150): [35, 105],
    }
)

########################
####### GT #############
########################
values["GT"] = {}
values["GT"]["M"] = RangeDict(
    {
        range(12): [2, 25],
        range(12, 18): [6, 40],
        range(18, 40): [10, 80],
        range(40, 150): [15, 115],
    }
)
values["GT"]["F"] = RangeDict(
    {
        range(12): [2, 25],
        range(12, 18): [6, 40],
        range(18, 40): [10, 45],
        range(40, 150): [10, 75],
    }
)

########################
####### Amylase ########
########################
values["Amylase"] = {}
values["Amylase"]["M"] = RangeDict({range(150): [25, 120],})
values["Amylase"]["F"] = values["Amylase"]["M"]

########################
####### Glukose ########
########################
values["Glukose"] = {}
values["Glukose"]["M"] = RangeDict({range(150): [4, 6],})
values["Glukose"]["F"] = values["Glukose"]["M"]


def nominal_values(Names, Sex, Age):
    global values

    # When a test is unknown default nominal values [0,inf] are returned
    missing = dict.fromkeys(["M", "F"], RangeDict({range(150): [0, np.inf]}))
    n_values = np.array([values.get(name, missing)[Sex][Age] for name in Names])

    return n_values


if __name__ == "__main__":

    with open("/data/Name_of_tests.txt", "r") as f:
        Names = [line.rstrip() for line in f]

    normal_values = nominal_values(Names, "M", 30)
    print(normal_values)
    print(nominal_values(["InventedTest1", "InventedTest2"], "F", 30))
