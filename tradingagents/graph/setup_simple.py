# TradingAgents/graph/setup_simple.py

from typing import Any, Dict
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, RemoveMessage

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState


def should_continue_market(state: AgentState) -> str:
    """判断市场分析是否需要继续（使用工具或完成）。"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools_market"
    return "Msg Clear Market"


def should_continue_social(state: AgentState) -> str:
    """判断社交媒体分析是否需要继续（使用工具或完成）。"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools_social"
    return "Msg Clear Social"


def should_continue_news(state: AgentState) -> str:
    """判断新闻分析是否需要继续（使用工具或完成）。"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools_news"
    return "Msg Clear News"


def should_continue_fundamentals(state: AgentState) -> str:
    """判断基本面分析是否需要继续（使用工具或完成）。"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools_fundamentals"
    return "Msg Clear Fundamentals"


def should_continue_debate(state: AgentState, max_rounds: int = 1) -> str:
    """判断投资辩论是否应该继续或进入研究经理。

    Args:
        state: 当前代理状态
        max_rounds: 最大辩论轮数

    Returns:
        "Research Manager"（辩论结束）或下一个辩论者
    """
    if state["investment_debate_state"]["count"] >= 2 * max_rounds:
        return "Research Manager"
    if state["investment_debate_state"]["current_response"].startswith("Bull"):
        return "Bear Researcher"
    return "Bull Researcher"


def should_continue_risk_analysis(state: AgentState, max_rounds: int = 1) -> str:
    """判断风险分析是否应该继续或进入投资组合经理。

    Args:
        state: 当前代理状态
        max_rounds: 最大讨论轮数

    Returns:
        "Portfolio Manager"（分析结束）或下一个风险分析师
    """
    if state["risk_debate_state"]["count"] >= 3 * max_rounds:
        return "Portfolio Manager"

    latest_speaker = state["risk_debate_state"]["latest_speaker"]
    if latest_speaker.startswith("Aggressive"):
        return "Conservative Analyst"
    elif latest_speaker.startswith("Conservative"):
        return "Neutral Analyst"
    return "Aggressive Analyst"


def _create_msg_delete():
    """创建消息删除节点，清空消息并添加占位符。

    Returns:
        一个删除消息的节点函数
    """
    def delete_messages(state):
        """清空消息并添加占位符（Anthropic兼容性）"""
        messages = state["messages"]

        # 删除所有消息
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # 添加一个最小的占位符消息
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        portfolio_manager_memory,
        max_debate_rounds: int = 1,
        max_risk_discuss_rounds: int = 1,
    ):
        """初始化图设置所需组件。

        Args:
            quick_thinking_llm: 用于快速分析任务的LLM
            deep_thinking_llm: 用于复杂推理任务的LLM
            tool_nodes: 每种分析师类型的工具节点字典
            bull_memory: 看涨研究员的记忆
            bear_memory: 看跌研究员的记忆
            trader_memory: 交易员的记忆
            invest_judge_memory: 研究经理的记忆
            portfolio_manager_memory: 投资组合经理的记忆
            max_debate_rounds: 看涨/看跌辩论的最大轮数（默认：1）
            max_risk_discuss_rounds: 风险分析的最大轮数（默认：1）
        """
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.portfolio_manager_memory = portfolio_manager_memory
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def _build_workflow_graph(
        self,
        market_analyst_node,
        social_analyst_node,
        news_analyst_node,
        fundamentals_analyst_node,
        market_delete_node,
        social_delete_node,
        news_delete_node,
        fundamentals_delete_node,
        market_tool_node,
        social_tool_node,
        news_tool_node,
        fundamentals_tool_node,
        bull_researcher_node,
        bear_researcher_node,
        research_manager_node,
        trader_node,
        aggressive_analyst,
        neutral_analyst,
        conservative_analyst,
        portfolio_manager_node,
    ):
        """构建工作流图，添加所有节点和边。

        这是一个独立的工作流创建函数，将所有节点添加到工作流中
        并定义它们之间的连接关系。

        Returns:
            StateGraph: 配置好的工作流图对象
        """
        # 创建工作流
        workflow = StateGraph(AgentState)

        # 显式添加分析师节点
        workflow.add_node("Market Analyst", market_analyst_node)
        workflow.add_node("Social Analyst", social_analyst_node)
        workflow.add_node("News Analyst", news_analyst_node)
        workflow.add_node("Fundamentals Analyst", fundamentals_analyst_node)

        # 显式添加删除节点
        workflow.add_node("Msg Clear Market", market_delete_node)
        workflow.add_node("Msg Clear Social", social_delete_node)
        workflow.add_node("Msg Clear News", news_delete_node)
        workflow.add_node("Msg Clear Fundamentals", fundamentals_delete_node)

        # 显式添加工具节点
        workflow.add_node("tools_market", market_tool_node)
        workflow.add_node("tools_social", social_tool_node)
        workflow.add_node("tools_news", news_tool_node)
        workflow.add_node("tools_fundamentals", fundamentals_tool_node)

        # 添加其他节点
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # 显式定义边（无循环，无条件语句）
        # 从市场分析师开始
        workflow.add_edge(START, "Market Analyst")

        # 连接市场分析师
        workflow.add_conditional_edges(
            "Market Analyst",
            should_continue_market,
            ["tools_market", "Msg Clear Market"],
        )
        workflow.add_edge("tools_market", "Market Analyst")
        workflow.add_edge("Msg Clear Market", "Social Analyst")

        # 连接社交媒体分析师
        workflow.add_conditional_edges(
            "Social Analyst",
            should_continue_social,
            ["tools_social", "Msg Clear Social"],
        )
        workflow.add_edge("tools_social", "Social Analyst")
        workflow.add_edge("Msg Clear Social", "News Analyst")

        # 连接新闻分析师
        workflow.add_conditional_edges(
            "News Analyst",
            should_continue_news,
            ["tools_news", "Msg Clear News"],
        )
        workflow.add_edge("tools_news", "News Analyst")
        workflow.add_edge("Msg Clear News", "Fundamentals Analyst")

        # 连接基本面分析师
        workflow.add_conditional_edges(
            "Fundamentals Analyst",
            should_continue_fundamentals,
            ["tools_fundamentals", "Msg Clear Fundamentals"],
        )
        workflow.add_edge("tools_fundamentals", "Fundamentals Analyst")
        workflow.add_edge("Msg Clear Fundamentals", "Bull Researcher")

        # 添加研究阶段边
        workflow.add_conditional_edges(
            "Bull Researcher",
            lambda state: should_continue_debate(state, self.max_debate_rounds),
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            lambda state: should_continue_debate(state, self.max_debate_rounds),
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")

        # 添加风险分析边
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            lambda state: should_continue_risk_analysis(state, self.max_risk_discuss_rounds),
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            lambda state: should_continue_risk_analysis(state, self.max_risk_discuss_rounds),
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            lambda state: should_continue_risk_analysis(state, self.max_risk_discuss_rounds),
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        return workflow

    def setup_graph(self):
        """设置并编译代理工作流图。

        创建包含所有4位分析师的固定图：市场、社交媒体、新闻、基本面。
        所有分析师按固定顺序包含在工作流中。

        Returns:
            Compiled: 编译好的工作流图，准备执行
        """
        # 步骤1：创建所有分析师节点
        market_analyst_node = create_market_analyst(self.quick_thinking_llm)
        social_analyst_node = create_social_media_analyst(self.quick_thinking_llm)
        news_analyst_node = create_news_analyst(self.quick_thinking_llm)
        fundamentals_analyst_node = create_fundamentals_analyst(self.quick_thinking_llm)

        # 步骤2：创建删除节点（内联实现）
        market_delete_node = _create_msg_delete()
        social_delete_node = _create_msg_delete()
        news_delete_node = _create_msg_delete()
        fundamentals_delete_node = _create_msg_delete()

        # 步骤3：获取工具节点
        market_tool_node = self.tool_nodes["market"]
        social_tool_node = self.tool_nodes["social"]
        news_tool_node = self.tool_nodes["news"]
        fundamentals_tool_node = self.tool_nodes["fundamentals"]

        # 步骤4：创建研究和管理节点
        bull_researcher_node = create_bull_researcher(
            self.quick_thinking_llm, self.bull_memory
        )
        bear_researcher_node = create_bear_researcher(
            self.quick_thinking_llm, self.bear_memory
        )
        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.invest_judge_memory
        )
        trader_node = create_trader(self.quick_thinking_llm, self.trader_memory)

        # 步骤5：创建风险分析节点
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(
            self.deep_thinking_llm, self.portfolio_manager_memory
        )

        # 步骤6：构建工作流图（提取的独立函数）
        workflow = self._build_workflow_graph(
            market_analyst_node=market_analyst_node,
            social_analyst_node=social_analyst_node,
            news_analyst_node=news_analyst_node,
            fundamentals_analyst_node=fundamentals_analyst_node,
            market_delete_node=market_delete_node,
            social_delete_node=social_delete_node,
            news_delete_node=news_delete_node,
            fundamentals_delete_node=fundamentals_delete_node,
            market_tool_node=market_tool_node,
            social_tool_node=social_tool_node,
            news_tool_node=news_tool_node,
            fundamentals_tool_node=fundamentals_tool_node,
            bull_researcher_node=bull_researcher_node,
            bear_researcher_node=bear_researcher_node,
            research_manager_node=research_manager_node,
            trader_node=trader_node,
            aggressive_analyst=aggressive_analyst,
            neutral_analyst=neutral_analyst,
            conservative_analyst=conservative_analyst,
            portfolio_manager_node=portfolio_manager_node,
        )

        # 步骤7：编译并返回
        return workflow.compile()
