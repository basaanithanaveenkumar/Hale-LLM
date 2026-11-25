"""Plain top-k router."""

from hale_llm.nn.moe.registry import register_router
from hale_llm.nn.moe.router import Router


@register_router("topk")
class BestKRouter(Router):
    """Send every token to the ``k`` experts with the highest gate score."""
