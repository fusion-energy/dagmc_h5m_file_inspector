# creates a dagmc geometry with material tags
# there is one material tag "material1", two "material2" and three "material3"
# the order is not sorted

import cadquery as cq
from cadquery import Assembly
from cad_to_dagmc import CadToDagmc


result1 = cq.Workplane().box(1, 1, 1)  # material1
result2 = cq.Workplane().moveTo(1, 0).box(1, 1, 1)  # material2
result3 = cq.Workplane().moveTo(2, 0).box(1, 1, 1)  # material3
result4 = cq.Workplane().moveTo(3, 0).box(1, 1, 1)  # material2
result5 = cq.Workplane().moveTo(4, 0).box(1, 1, 1)  # material3
result6 = cq.Workplane().moveTo(5, 0).box(1, 1, 1)  # material3

assembly = Assembly()
assembly.add(result1)
assembly.add(result2)
assembly.add(result3)
assembly.add(result4)
assembly.add(result5)
assembly.add(result6)

my_model = CadToDagmc()
my_model.add_cadquery_object(
    assembly,
    material_tags=[
        "material1",
        "material2",
        "material3",
        "material2",
        "material3",
        "material3",
    ],
)
my_model.export_dagmc_h5m_file("dagmc.h5m", min_mesh_size=0.5, max_mesh_size=1.0)
