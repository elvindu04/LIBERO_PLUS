
from dataclasses import dataclass, field

from typing import List, Union, Tuple, Optional


@dataclass
class Distribution:
    centroid: List[float]  # (x_min, y_min, x_max, y_max) in bddl
    loc_bounds: List[float]
    # rot: float = 0.0  # TODO: should we change the rotation to mutliple axes? currenlty LIBERO only supports yaw rotation
    rot_bounds: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))


@dataclass
class ReceptacleInitState:
    name: Union[str, List[str]]
    init_state: Distribution


# TODO: do we need to handle duplicates of objects?
@dataclass
class ObjectInitState:
    name: Union[str, List[str]]
    init_state: Distribution
    target_name: Optional[Union[str, List[str]]] = None


@dataclass
class BackgroundInitState:
    name: Union[str, List[str]]
    # TODO: can we also modify lighting or other attributes?


@dataclass
class TableInitState:
    name: Union[str, List[str]]
    init_state: Distribution


@dataclass
class GrabberInitState:
    name: Union[str, List[str]]
    # TODO: any other modifications?


@dataclass
class TaskInitState:
    receptacle: List[ReceptacleInitState]
    manipulated_obj: List[ObjectInitState]
    distractor_obj: List[ObjectInitState]
    background: BackgroundInitState
    # table: TableInitState
    # grabber: GrabberInitState


# TODO: do we need to add host (named :target in bddl) object/location in the config?


# for LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl

receptacle_locations = {
    "left": {
        "centroid": [0.02, 0.23],
        "loc_bounds": [0.18, 0.05],
        "obj_centroid": [-0.025, 0.06],
        "obj_loc_bounds": [0.175, 0.24],

    },
    "right": {
        "centroid": [0.02, -0.23],
        "loc_bounds": [0.18, 0.05],
        "obj_centroid": [-0.025, -0.06],
        "obj_loc_bounds": [0.175, 0.24],
    },
    "back": {
        "centroid": [-0.15, 0.0],
        "loc_bounds": [0.05, 0.23],
        "obj_centroid": [0.025, 0.0],
        "obj_loc_bounds": [0.125, 0.3],
    }
}

basket = ReceptacleInitState(
    name="basket",
    init_state=Distribution(
        # # back
        # centroid=[-0.15, 0.0],
        # loc_bounds=[0.05, 0.23]

        # test
        centroid=[-0.0, 0.0],
        loc_bounds=[0.005, 0.005]

        # # left
        # centroid=[0.02, 0.23],
        # loc_bounds=[0.18, 0.05]
        # # right
        # centroid=[0.02, -0.23],
        # loc_bounds=[0.18, 0.05]
    )
)

milk = ObjectInitState(
    name="milk",
    init_state=Distribution(
        centroid=[0.0, -0.15],
        loc_bounds=[0.005, 0.005]

        # centroid=[0.05, -0.1],
        # loc_bounds=[0.025, 0.025]
    )
)


cream_cheese = ObjectInitState(
    name="cream_cheese",
    init_state=Distribution(
        centroid=[0.1, -0.2],
        loc_bounds=[0.025, 0.025]
    )
)


orange_juice = ObjectInitState(
    name="orange_juice",
    init_state=Distribution(
        centroid=[0.0, -0.25],
        loc_bounds=[0.025, 0.025]
    )
)


tomato_sauce = ObjectInitState(
    name="tomato_sauce",
    init_state=Distribution(
        centroid=[-0.1, 0.5],
        loc_bounds=[0.025, 0.025]
    ),
    target_name="basket"
)



alphabet_soup = ObjectInitState(
    name="alphabet_soup",
    init_state=Distribution(
        centroid=[-0.1, -0.15],
        loc_bounds=[0.025, 0.025]
    ),
    target_name="basket"
)



butter = ObjectInitState(
    name="butter",
    init_state=Distribution(
        centroid=[0.05, 0.05],
        loc_bounds=[0.025, 0.025]
    )
)


ketchup = ObjectInitState(
    name="ketchup",
    init_state=Distribution(
        centroid=[-0.25, -0.15],
        loc_bounds=[0.025, 0.025]
    )
)

task = TaskInitState(
    receptacle=[basket],
    # manipulated_obj=[alphabet_soup],
    distractor_obj=[milk],
    manipulated_obj=[alphabet_soup, tomato_sauce],
    # distractor_obj=[milk, cream_cheese, orange_juice, butter, ketchup],
    background=BackgroundInitState(name="living_room"),
)


import numpy as np
from libero.libero.envs.objects import get_object_dict, get_object_fn
from libero.libero.envs.predicates import get_predicate_fn_dict, get_predicate_fn
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info

def generate_task_bddl(init_state: TaskInitState, task_id: int):
    
    target_objs = sorted([obj.name for obj in init_state.manipulated_obj])  # sorted to avoid aliasing
    scene_name = f"{init_state.background.name}_{'_'.join(target_objs)}_scene{task_id}"


    class_name = f"{init_state.background.name.capitalize()}{''.join([obj.capitalize() for obj in target_objs])}Scene{task_id}"

    target_objs = [obj.replace("_", " ") for obj in target_objs]
    target_phrases = []

    for obj in init_state.manipulated_obj:
        phrase = f"put the {obj.name.replace('_', ' ')} in the {obj.target_name}"
        target_phrases.append(phrase)


    language = ", ".join(target_phrases[:-1]) + " and " + target_phrases[-1]


    workspace = init_state.background.name + "_table"

    @register_mu(scene_type=init_state.background.name, name_override=class_name)
    class DummyClass(InitialSceneTemplates):
        def __init__(self):
            fixture_num_info = {
                workspace: 1,
            }

            object_num_info = {}

            self.obj_list = init_state.receptacle + init_state.manipulated_obj + init_state.distractor_obj

            for obj in self.obj_list:
                object_num_info[obj.name] = 1

            super().__init__(
                workspace_name=workspace,
                fixture_num_info=fixture_num_info,
                object_num_info=object_num_info
            )

        def define_regions(self):

            for obj in self.obj_list:
                self.regions.update(
                    self.get_region_dict(region_centroid_xy=obj.init_state.centroid,
                                        region_name=f"{obj.name}_init_region", 
                                        target_name=self.workspace_name, 
                                        region_half_width=obj.init_state.loc_bounds[0],
                                        region_half_len=obj.init_state.loc_bounds[1],
                                        yaw_rotation=obj.init_state.rot_bounds)
                )
            
            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

        @property
        def init_states(self):
            states = []
            
            for obj in self.obj_list:
                obj_tup = ("On", f"{obj.name}_1", f"{workspace}_{obj.name}_init_region")
                states.append(obj_tup)
            
            return states
        

    goal_states = []

    for obj in init_state.manipulated_obj:
        obj_tup = ("In", f"{obj.name}_1", f"{obj.target_name}_1_contain_region")
        goal_states.append(obj_tup)

    register_task_info(language,
                       scene_name=scene_name,
                       objects_of_interest=[obj.name + "_1" for obj in init_state.manipulated_obj],
                       goal_states=goal_states
    )

    YOUR_BDDL_FILE_PATH = "/home/leisongao/LIBERO/libero/libero/bddl_files/libero_10_mod"
    bddl_file_names, failures = generate_bddl_from_task_info(folder=YOUR_BDDL_FILE_PATH)

    print(bddl_file_names)

    print("Encountered some failures: ", failures)


generate_task_bddl(task, 0)