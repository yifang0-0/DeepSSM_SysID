from models.model_phy import kuka300
from models.model_phy import toy_lgssm
class phy_model():
    def __init__(self, phy_type):
        self.phy_type = phy_type
        if phy_type == 'kuka300':
            self.model = kuka300()
        elif phy_type == 'toy_lgssm':
            # decide how to change or where to add the congifuration that what parts of the models are available (do I need seperated model for that or maybe, no)
            self.model == toy_lgssm()# can be initialed by adding A,B,C,D matrix here
            
            