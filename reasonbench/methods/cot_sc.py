import random
import logging
import asyncio
from typing import TypedDict
from collections import Counter
from omegaconf import OmegaConf
from ..typedefs import Method, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
from .. import MethodFactory, AgentDictFactory
from ..utils import Resampler
logger = logging.getLogger(__name__)

@AgentDictFactory.register
class AgentDictCoT(TypedDict):
    step: Agent # ActAgent
    step_params: DecodingParameters


@MethodFactory.register
class MethodCOT_SC(Method):
    def __init__(self,
                 model: Model,
                 agents: AgentDictCoT,
                 env: Environment,
                 config: OmegaConf,
                ):
        super().__init__(model, agents, env, config)
        
        self.step_agent = agents["step"]
        self.step_params = agents["step_params"]

        self.n = config.n
        assert self.n > 1, "CoT-SC needs at least 2 outputs"
        

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict=None):
        randomness = idx
        random.seed(randomness)

        actions = await self.step_agent.act(
            model=self.model,
            state=state,
            n=self.n,
            namespace=namespace,
            request_id=f"idx{idx}-{hash(state)}",
            params=self.step_params
        )

        votes = [action for action in actions]
        counts = Counter(votes)
        most_common_action = counts.most_common(1)[0][0]
        state = self.env.step(state, most_common_action)
        return [state]

    