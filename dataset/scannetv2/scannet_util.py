g_label_names = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator', 'picture', 'cabinet', 'otherfurniture']

def get_raw2scannetv2_label_map():
    lines = [line.rstrip() for line in open('scannetv2-labels.combined.tsv')]
    lines_0 = lines[0].split('\t')
    print(lines_0)
    print(len(lines))
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(g_label_names)
        elements = lines[i].split('\t')
        raw_name = elements[1]
        if (elements[1] != elements[2]):
            print('{}: {} {}'.format(i, elements[1], elements[2]))
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = 'unannotated'
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet

g_raw2scannetv2 = get_raw2scannetv2_label_map()