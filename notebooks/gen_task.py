import numpy as np
from libero.libero.envs.objects import get_object_dict, get_object_fn
from libero.libero.envs.predicates import get_predicate_fn_dict, get_predicate_fn
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info

from dataclasses import dataclass


@dataclass
class InitState:
    obj: str
    workspace: str
    task_id: int
    basket_loc: list[float]
    object_loc: list[float]


def generate_task_bddl(object, workspace, task_id, basket_loc, object_loc):
    scene_name = f"{workspace}_{object}_scene{task_id}"
    language = f"put the {object.replace('_', ' ')} in the receptacle"

    workspace_temp = workspace
    workspace += "_table"
    
    @register_mu(scene_type=workspace_temp, name_override=f"{workspace_temp.capitalize()}{object.capitalize()}Scene{task_id}")
    class DummyClass(InitialSceneTemplates):
        def __init__(self):
            fixture_num_info = {
                workspace: 1,
            }

            object_num_info = {
                object : 1,
                "cream_cheese" : 1,
                "basket": 1,
            }

            super().__init__(
                workspace_name=workspace,
                fixture_num_info=fixture_num_info,
                object_num_info=object_num_info
            )

        def define_regions(self):
            self.regions.update(
                self.get_region_dict(region_centroid_xy=basket_loc, 
                                    #  region_centroid_xy=[0.0, -0.30], 
                                    region_name="basket_init_region", 
                                    target_name=self.workspace_name, 
                                    region_half_len=0.01,
                                    yaw_rotation=(0.0, 0.0))
            )

            self.regions.update(
                self.get_region_dict(region_centroid_xy=object_loc, 
                                    #  region_centroid_xy=[0.0, 0.03], 
                                    region_name=f"{object}_init_region", 
                                    target_name=self.workspace_name, 
                                    region_half_len=0.15)
            )

            self.regions.update(
                self.get_region_dict(region_centroid_xy=object_loc, 
                                    #  region_centroid_xy=[0.0, 0.03], 
                                    region_name=f"cream_cheese_init_region", 
                                    target_name=self.workspace_name, 
                                    region_half_len=0.15)
            )

            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

        @property
        def init_states(self):
            states = [
                ("On", f"{object}_1", f"{workspace}_{object}_init_region"),
                ("On", f"cream_cheese_1", f"{workspace}_cream_cheese_init_region"),
                ("On", "basket_1", f"{workspace}_basket_init_region")
                ]
            return states
        

    register_task_info(language,
                        scene_name=scene_name,
                        objects_of_interest=[f"{object}_1", "basket_1"],
                        goal_states=[("In", f"{object}_1", "basket_1_contain_region")]
    )

    YOUR_BDDL_FILE_PATH = "/home/leisongao/LIBERO/libero/libero/bddl_files/libero_10_diff_obj"
    bddl_file_names, failures = generate_bddl_from_task_info(folder=YOUR_BDDL_FILE_PATH)

    print(bddl_file_names)

    print("Encountered some failures: ", failures)



if __name__=="__main__":
    # init_states = [
    #     InitState("butter", "living_room", 1, [-0.05, 0.27], [0.0, 0.0]),
    #     InitState("butter", "living_room", 2, [-0.10, 0.27], [0.0, 0.0]),
    #     InitState("butter", "living_room", 3, [-0.15, 0.27], [0.0, 0.0]),
    #     InitState("butter", "living_room", 4, [0.05, 0.27], [0.0, 0.0]),
    #     InitState("butter", "living_room", 5, [0.10, 0.27], [0.0, 0.0]),
    #     InitState("butter", "living_room", 6, [0.15, 0.27], [0.0, 0.0]), ### too far forward for action unnorm
    #     InitState("butter", "living_room", 7, [0.00, 0.27], [0.0, 0.0]),
    # ]


    obj = "alphabet_soup"
    loc = "coffee"

    init_states = [
        [[-0.05, 0.17], [-0.05, -0.15]],
        # [[-0.10, 0.17], [-0.05, -0.15]],
        # [[-0.15, 0.17], [-0.05, -0.15]],
        # [[-0.20, 0.17], [-0.05, -0.15]],
        # [[0.00,  0.17], [-0.05, -0.15]],
        # [[0.05,  0.17], [-0.05, -0.15]],
        # [[0.10,  0.17], [-0.05, -0.15]],
    ]
    

    init_states_list = []
    offset = 0
    for i, state in enumerate(init_states):
        init_states_list.append(InitState(obj, loc, i + offset, state[0], state[1]))

    
    for state in init_states_list:
        generate_task_bddl(state.obj, state.workspace, state.task_id, state.basket_loc, state.object_loc)