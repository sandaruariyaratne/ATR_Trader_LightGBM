# Lazy imports — individual agents import ccxt/ccxt.pro which is only
# needed at runtime. Tests import agents directly to avoid the exchange dep.

__all__ = [
    "MarketDataAgent",
    "FeatureEngineeringAgent",
    "DecisionAgent",
    "RiskManagementAgent",
    "ExecutionAgent",
]


def __getattr__(name: str):
    if name == "MarketDataAgent":
        from .market_data_agent import MarketDataAgent
        return MarketDataAgent
    if name == "FeatureEngineeringAgent":
        from .feature_engineering_agent import FeatureEngineeringAgent
        return FeatureEngineeringAgent
    if name == "DecisionAgent":
        from .decision_agent import DecisionAgent
        return DecisionAgent
    if name == "RiskManagementAgent":
        from .risk_management_agent import RiskManagementAgent
        return RiskManagementAgent
    if name == "ExecutionAgent":
        from .execution_agent import ExecutionAgent
        return ExecutionAgent
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
