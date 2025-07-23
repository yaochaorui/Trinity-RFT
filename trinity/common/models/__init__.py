from collections import defaultdict
from typing import List, Tuple

from trinity.common.config import Config
from trinity.common.models.model import InferenceModel
from trinity.utils.log import get_logger


class _BundleAllocator:
    """An allocator for bundles."""

    def __init__(self, node_bundle_map: dict[str, list]) -> None:
        self.logger = get_logger(__name__)
        self.node_bundle_list = [value for value in node_bundle_map.values()]
        self.node_list = [key for key in node_bundle_map.keys()]
        self.nid = 0
        self.bid = 0

    def allocate(self, num: int) -> list:
        # allocate num bundles from current node
        if self.bid + num > len(self.node_bundle_list[self.nid]):
            raise ValueError(
                "Bundle allocation error, a tensor parallel group"
                " is allocated across multiple nodes."
            )
        bundle_list = self.node_bundle_list[self.nid][self.bid : self.bid + num]
        self.logger.info(f"Allocate bundles {bundle_list} on node {self.node_list[self.nid]}.")
        self.bid += num
        if self.bid == len(self.node_bundle_list[self.nid]):
            self.bid = 0
            self.nid += 1
        return bundle_list


def create_inference_models(
    config: Config,
) -> Tuple[List[InferenceModel], List[List[InferenceModel]]]:
    """Create `engine_num` rollout models.

    Each model has `tensor_parallel_size` workers.
    """
    import ray
    from ray.util.placement_group import placement_group, placement_group_table
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from trinity.common.models.vllm_model import vLLMRolloutModel

    logger = get_logger(__name__)
    engine_num = config.explorer.rollout_model.engine_num
    tensor_parallel_size = config.explorer.rollout_model.tensor_parallel_size

    rollout_engines = []
    if config.explorer.rollout_model.engine_type.startswith("vllm"):
        engine_cls = vLLMRolloutModel
    else:
        raise ValueError(f"Unknown engine type: {config.explorer.rollout_model.engine_type}")

    main_bundles = [{"GPU": 1} for _ in range(engine_num * tensor_parallel_size)]
    auxiliary_bundles = [
        {"GPU": 1}
        for _ in range(
            sum(
                [
                    model.engine_num * model.tensor_parallel_size
                    for model in config.explorer.auxiliary_models
                ]
            )
        )
    ]
    pg = placement_group(main_bundles + auxiliary_bundles, strategy="PACK")
    ray.get(pg.ready())

    rollout_engines = []
    auxiliary_engines = []

    # to address https://github.com/ray-project/ray/issues/51117
    # aggregate bundles belonging to the same node
    bundle_node_map = placement_group_table(pg)["bundles_to_node_id"]
    node_bundle_map = defaultdict(list)
    for bundle_id, node_id in bundle_node_map.items():
        node_bundle_map[node_id].append(bundle_id)
    allocator = _BundleAllocator(node_bundle_map)
    namespace = ray.get_runtime_context().namespace
    # create rollout models
    for _ in range(config.explorer.rollout_model.engine_num):
        bundles_for_engine = allocator.allocate(config.explorer.rollout_model.tensor_parallel_size)
        config.explorer.rollout_model.bundle_indices = ",".join(
            [str(bid) for bid in bundles_for_engine]
        )
        rollout_engines.append(
            ray.remote(engine_cls)
            .options(
                num_cpus=0,
                num_gpus=0 if config.explorer.rollout_model.tensor_parallel_size > 1 else 1,
                namespace=namespace,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundles_for_engine[0],
                ),
            )
            .remote(
                config=config.explorer.rollout_model,
            )
        )
    if config.explorer.rollout_model.enable_openai_api:
        for engine in rollout_engines:
            engine.run_api_server.remote()
    if config.explorer.rollout_model.enable_history:
        logger.info(
            "Model History recording is enabled. Please periodically extract "
            "history via `extract_experience_from_history` to avoid out-of-memory issues."
        )
    # create auxiliary models
    for model_config in config.explorer.auxiliary_models:
        engines = []
        for _ in range(model_config.engine_num):
            bundles_for_engine = allocator.allocate(model_config.tensor_parallel_size)
            model_config.enable_openai_api = True
            model_config.engine_type = "vllm"
            model_config.bundle_indices = ",".join([str(bid) for bid in bundles_for_engine])
            engines.append(
                ray.remote(vLLMRolloutModel)
                .options(
                    num_cpus=0,
                    num_gpus=0 if model_config.tensor_parallel_size > 1 else 1,
                    namespace=namespace,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_capture_child_tasks=True,
                        placement_group_bundle_index=bundles_for_engine[0],
                    ),
                )
                .remote(config=model_config)
            )
        auxiliary_engines.append(engines)
    # all auxiliary engines run api server
    for engines in auxiliary_engines:
        for engine in engines:
            engine.run_api_server.remote()

    return rollout_engines, auxiliary_engines
