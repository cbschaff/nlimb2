from isaacgym import gymapi
import gin


@gin.configurable
class SimParams(gymapi.SimParams):
    def __init__(self,
                 dt: float = 1. / 60.,
                 substeps: int = 2,
                 gravity: float = -9.81,
                 up_axis: str = 'z',
                 num_client_threads: int = 0,
                 use_gpu_pipeline: bool = True):
        gymapi.SimParams.__init__(self)
        self.dt = dt
        self.substeps = substeps

        if up_axis == 'z':
            self.up_axis = gymapi.UP_AXIS_Z
            self.gravity.x = 0
            self.gravity.y = 0
            self.gravity.z = gravity
        elif up_axis == 'y':
            self.up_axis = gymapi.UP_AXIS_Y
            self.gravity.x = 0
            self.gravity.y = gravity
            self.gravity.z = 0
        else:
            raise ValueError(f"Invalid Up Axis: {up_axis}")

        self.num_client_threads = num_client_threads
        self.use_gpu_pipeline = use_gpu_pipeline
        self.physx = PhysXParams()
        self.flex = FlexParams()


@gin.configurable
class PhysXParams(gymapi.PhysXParams):
    def __init__(self,
                 bounce_threshold_velocity: float = 0.2,
                 contact_offset: float = 0.01,
                 max_depenetration_velocity: float = 10.0,
                 num_position_iterations: int = 4,
                 num_threads: int = 4,
                 num_velocity_iterations: int = 4,
                 rest_offset: float = 0.0,
                 solver_type: int = 1,  # 0: pgs, 1: tgs
                 use_gpu: bool = True,
                 contact_collection: int = 0,  # 0: never, 1: at last substep, 2: always
                 default_buffer_size_multiplier: float = 5.0,
                 num_subscenes: int = 4,
                 ):
        gymapi.PhysXParams.__init__(self)
        self.bounce_threshold_velocity = bounce_threshold_velocity
        self.contact_offset = contact_offset
        self.max_depenetration_velocity = max_depenetration_velocity
        self.num_position_iterations = num_position_iterations
        self.num_threads = num_threads
        self.num_velocity_iterations = num_velocity_iterations
        self.rest_offset = rest_offset
        self.solver_type = solver_type
        self.use_gpu = use_gpu
        self.contact_collection = [gymapi.ContactCollection.CC_NEVER,
                                   gymapi.ContactCollection.CC_LAST_SUBSTEP,
                                   gymapi.ContactCollection.CC_ALL_SUBSTEPS][contact_collection]
        self.default_buffer_size_multiplier = default_buffer_size_multiplier
        self.num_subscenes = num_subscenes



@gin.configurable
class FlexParams(gymapi.FlexParams):
    def __init__(self,
                 num_inner_iterations: int = 15,
                 num_outer_iterations: int = 4,
                 relaxation: float = 0.75,
                 solver_type: int = 5,
                 warm_start: float = 0.4,
                ):
        self.num_inner_iterations = num_inner_iterations
        self.num_outer_iterations = num_outer_iterations
        self.relaxation = relaxation
        self.solver_type = solver_type
        self.warm_start = warm_start
