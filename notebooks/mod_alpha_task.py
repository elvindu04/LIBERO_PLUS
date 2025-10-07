
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


    if init_state.background.name == "floor":
        workspace = "floor"
    else:
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

    YOUR_BDDL_FILE_PATH = "/home/leisongao/LIBERO/libero/libero/bddl_files/libero_10_diff_obj"
    bddl_file_names, failures = generate_bddl_from_task_info(folder=YOUR_BDDL_FILE_PATH)

    print(bddl_file_names)

    print("Encountered some failures: ", failures)




# we create combinations of all the dimensions of complexity
# instead maintain mapping between id and the task complexity

# TODO: is it easier to spexify corners of bbox distribution instead of centroid and xy? maybe change impl
def generate_init_class(data, add_rot, background):
    
    if add_rot:
        rot_bounds=(-np.pi, np.pi)
    else:
        rot_bounds=(0, 0)


    basket = ReceptacleInitState(
        name="basket",
        init_state=Distribution(
            centroid=data["centroid"],
            loc_bounds=data["loc_bounds"],
            rot_bounds=rot_bounds
        )
    )

    manipulated_obj = []
    manipulated_obj_names = ["alphabet_soup", "tomato_sauce"]
    distractor_obj = []
    distractor_obj_names = ["milk", "cream_cheese", "orange_juice", "butter", "ketchup"]

    for name in manipulated_obj_names:
        obj = ObjectInitState(
            name=name,
            init_state=Distribution(
                centroid=data["obj_centroid"],
                loc_bounds=data["obj_loc_bounds"],
                rot_bounds=rot_bounds
            ),
            target_name="basket"
        )
        manipulated_obj.append(obj)

    
    for name in distractor_obj_names:
        obj = ObjectInitState(
            name=name,
            init_state=Distribution(
                centroid=data["obj_centroid"],
                loc_bounds=data["obj_loc_bounds"],
                rot_bounds=rot_bounds
            ),
        )
        distractor_obj.append(obj)

    task = TaskInitState(
        receptacle=[basket],
        manipulated_obj=manipulated_obj,
        distractor_obj=distractor_obj,
        background=BackgroundInitState(name=background),
    )

    return task

receptacle_locations = {
    "receptacle_left": {
        "centroid": [0.02, 0.23],  # 0.08 basket buffer
        "loc_bounds": [0.18, 0.05],
        "obj_centroid": [-0.025, -0.1],
        "obj_loc_bounds": [0.175, 0.2],

    },
    "receptacle_right": {
        "centroid": [0.02, -0.23],
        "loc_bounds": [0.18, 0.05],
        "obj_centroid": [-0.025, 0.1],
        "obj_loc_bounds": [0.175, 0.2],
    },
    "receptacle_back": {
        "centroid": [-0.15, 0.0],  # 0.1 basket buffer
        "loc_bounds": [0.05, 0.23],
        "obj_centroid": [0.075, 0.0],
        "obj_loc_bounds": [0.075, 0.3],
    },
    "receptacle_front": {
        "centroid": [0.1, 0.0],
        "loc_bounds": [0.05, 0.23],
        "obj_centroid": [-0.125, 0.0],
        "obj_loc_bounds": [0.075, 0.3],
    }
}

backgrounds = ["living_room", "kitchen", "study", "floor"] # should we add Coffee?

id_task_mapping = {}

for i, background in enumerate(backgrounds):
    for j, add_rot in enumerate([False]): # TODO: add rotation, fix bug in LIBERO
        for k, (task_name, data) in enumerate(receptacle_locations.items()):

            task_id =  10000 + i * 1000 + j * 100 + k 
            # we create task id 10_000 is for the current libero task, thousands are for background, hundreds is to indicate rot, k is receptacle loc

            id_task_mapping[task_id] = f"{background}_rot_{add_rot}_{task_name}"


            task = generate_init_class(data=data, add_rot=add_rot, background=background)


            generate_task_bddl(task, task_id)


        print(f"\nTask ID mapping:\n{id_task_mapping}")

