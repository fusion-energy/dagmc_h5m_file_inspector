
import numpy as np
import pymoab as mb
from pymoab import core, types

def load_moab_file(filename: str):
    """Loads a h5m into a Moab Core object and returns the object"""
    moab_core = core.Core()
    moab_core.load_file(filename)
    return moab_core


# def find_volume_ids_in_h5m(filename: Optional[str] = "dagmc.h5m") -> List[str]:
#     """Reads in a DAGMC h5m file and uses PyMoab to find the volume ids of the
#     volumes in the file
#     Arguments:
#         filename:
#     Returns:
#         The filename of the h5m file created
#     """

#     # create a new PyMOAB instance and load the specified DAGMC file
#     moab_core = load_moab_file(filename)

#     # retrieve the category tag on the instance
#     cat_tag = moab_core.tag_get_handle(types.CATEGORY_TAG_NAME)

#     # get the set of entities using the provided category tag name
#     # (0 means search on the instance's root set)
#     ents = moab_core.get_entities_by_type_and_tag(
#         0, types.MBENTITYSET, [cat_tag], ["Volume"]
#     )

#     # retrieve the IDs of the entities
#     ids = moab_core.tag_get_data(cat_tag, ents).flatten()

#     return sorted(list(ids))

def get_groups(mbcore):
  
  category_tag = mbcore.tag_get_handle(mb.types.CATEGORY_TAG_NAME)
  
  group_category = np.array(["Group"])
  
  group_ents = mbcore.get_entities_by_type_and_tag(0, mb.types.MBENTITYSET, category_tag, group_category)
  
  return group_ents

    
def get_vol_mat_map(group_ents, mbcore):  
  name_tag = mbcore.tag_get_handle(mb.types.NAME_TAG_NAME)
  id_tag = mbcore.tag_get_handle(mb.types.GLOBAL_ID_TAG_NAME)
  vol_mat = {}

  
  for group_ent in group_ents:

    group_name = mbcore.tag_get_data(name_tag, group_ent)[0][0]
    # optionally confirm that this is a material!

    if group_name.startswith('mat:'):

      print('group_ent',group_ent)
      print('group_name',group_name)

      vols = mbcore.get_entities_by_type(group_ent, mb.types.MBENTITYSET)

      print('vols', vols)

      for vol in vols:
          id = mbcore.tag_get_data(id_tag, vol)[0][0]
          vol_mat[id] = group_name

  return vol_mat

def get_volumes_and_materials_from_h5m(filename):
    mbcore = load_moab_file(filename)
    group_ents = get_groups(mbcore)
    vol_mat = get_vol_mat_map(group_ents, mbcore)
    return vol_mat
