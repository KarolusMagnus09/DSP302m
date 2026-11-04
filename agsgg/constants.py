"""Hằng số Action Genome (36 lớp thực thể, 26 vị từ chia 3 nhóm)."""

OBJECTS = [
    'person', 'bag', 'bed', 'blanket', 'book', 'box', 'broom', 'chair',
    'closet/cabinet', 'clothes', 'cup/glass/bottle', 'dish', 'door', 'doorknob',
    'doorway', 'floor', 'food', 'groceries', 'laptop', 'light', 'medicine',
    'mirror', 'paper/notebook', 'phone/camera', 'picture', 'pillow',
    'refrigerator', 'sandwich', 'shelf', 'shoe', 'sofa/couch', 'table',
    'television', 'towel', 'vacuum', 'window',
]
# Chủ thể luôn là person -> bộ phân loại vật chỉ có 35 lớp (bỏ person), chỉ số 0..34
OBJ_CLASSES = OBJECTS[1:]
N_OBJ = len(OBJ_CLASSES)  # 35

ATTENTION = ['looking_at', 'not_looking_at', 'unsure']
SPATIAL = ['above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in']
CONTACTING = [
    'carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back',
    'holding', 'leaning_on', 'lying_on', 'not_contacting', 'other_relationship',
    'sitting_on', 'standing_on', 'touching', 'twisting', 'wearing', 'wiping',
    'writing_on',
]
N_ATT, N_SPA, N_CON = len(ATTENTION), len(SPATIAL), len(CONTACTING)  # 3, 6, 17
PREDICATES = ATTENTION + SPATIAL + CONTACTING  # 26
GROUP_SLICES = [(0, N_ATT), (N_ATT, N_ATT + N_SPA), (N_ATT + N_SPA, N_ATT + N_SPA + N_CON)]
assert len(OBJECTS) == 36 and len(PREDICATES) == 26
