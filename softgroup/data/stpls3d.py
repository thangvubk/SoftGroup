from .custom import CustomDataset


class STPLS3DDataset(CustomDataset):

    CLASSES = ('building', 'low vegetation', 'med. vegetation', 'high vegetation', 'vehicle',
               'truck', 'aircraft', 'militaryVehicle', 'bike', 'motorcycle', 'light pole',
               'street sign', 'clutter', 'fence')

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        ret = super().getInstanceInfo(xyz, instance_label, semantic_label)
        instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
        # ignore instance of class 0 and reorder class id
        instance_cls = [x - 1 if x != -100 else x for x in instance_cls]
        return instance_num, instance_pointnum, instance_cls, pt_offset_label
