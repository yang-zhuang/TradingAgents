"""
Simplified TradingAgents Analysis Runner

This is a simplified version of cli/main.py that:
- Removes all UI code (rich/typer)
- Removes interactive prompts
- Uses trading_graph_simple.py with fixed LLM configuration (Qwen3-14B)
- Uses fixed analyst set (all 4 analysts: market, social, news, fundamentals)
- Returns final_state for programmatic use

Dependencies:
- tradingagents.graph.trading_graph_simple: Simplified graph with fixed LLM
- tradingagents.graph.setup_simple: Simplified graph setup with inline logic
"""

from typing import Optional
import datetime
from pathlib import Path
from collections import deque
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from tradingagents.graph.trading_graph_simple import TradingAgentsGraph
from cli.models import AnalystType
from cli.stats_handler import StatsCallbackHandler

# Default configuration for simplified analysis
DEFAULT_CONFIG_VALUES = {
    "ticker": "SPY",
    "analysis_date": datetime.datetime.now().strftime("%Y-%m-%d"),
    "analysts": [  # 保留用于兼容性，但实际不使用（图是固定的）
        AnalystType.MARKET,
        AnalystType.SOCIAL,
        AnalystType.NEWS,
        AnalystType.FUNDAMENTALS
    ],
    "research_depth": 1,  # 与 SIMPLE_CONFIG 中的 max_debate_rounds 默认值一致
    "debug": False,  # 默认不开启调试模式
}


class MessageBuffer:
    """Simplified buffer for tracking analysis progress without UI."""

    # Fixed teams that always run (not user-selectable)
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Analyst name mapping
    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "social": "Social Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # Report section mapping: section -> (analyst_key for filtering, finalizing_agent)
    REPORT_SECTIONS = {
        "market_report": ("market", "Market Analyst"),
        "sentiment_report": ("social", "Social Analyst"),
        "news_report": ("news", "News Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "investment_plan": (None, "Research Manager"),
        "trader_investment_plan": (None, "Trader"),
        "final_trade_decision": (None, "Portfolio Manager"),
    }

    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.agent_status = {}
        self.report_sections = {}
        self.selected_analysts = []
        self._last_message_id = None

    def init_for_analysis(self, selected_analysts):
        """Initialize agent status and report sections based on selected analysts."""
        self.selected_analysts = [a.lower() for a in selected_analysts]

        # Build agent_status dynamically
        self.agent_status = {}

        # Add selected analysts
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # Add fixed teams
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # Build report_sections dynamically
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # Reset other state
        self.messages.clear()
        self.tool_calls.clear()
        self._last_message_id = None

    def add_message(self, message_type, content):
        """Add a message to the buffer (for file logging)."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        """Add a tool call to the buffer (for file logging)."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        """Update agent status."""
        if agent in self.agent_status:
            self.agent_status[agent] = status

    def update_report_section(self, section_name, content):
        """Update a report section."""
        if section_name in self.report_sections:
            self.report_sections[section_name] = content


def update_analyst_statuses(message_buffer, chunk):
    """Update analyst statuses based on accumulated report state (no UI)."""
    selected = message_buffer.selected_analysts
    found_active = False

    # Report key mapping
    report_keys = {
        "market": "market_report",
        "social": "sentiment_report",
        "news": "news_report",
        "fundamentals": "fundamentals_report",
    }

    for analyst_key in ["market", "social", "news", "fundamentals"]:
        if analyst_key not in selected:
            continue

        agent_name = message_buffer.ANALYST_MAPPING[analyst_key]
        report_key = report_keys[analyst_key]

        # Capture new report content from current chunk
        if chunk.get(report_key):
            message_buffer.update_report_section(report_key, chunk[report_key])

        # Determine status from accumulated sections, not just current chunk
        has_report = bool(message_buffer.report_sections.get(report_key))

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
        elif not found_active:
            message_buffer.update_agent_status(agent_name, "in_progress")
            found_active = True
        else:
            message_buffer.update_agent_status(agent_name, "pending")

    # When all analysts complete, transition research team to in_progress
    if not found_active and selected:
        if message_buffer.agent_status.get("Bull Researcher") == "pending":
            message_buffer.update_agent_status("Bull Researcher", "in_progress")


def extract_content_string(content):
    """Extract string content from various message formats."""
    import ast

    def is_empty(val):
        """Check if value is empty using Python's truthiness."""
        if val is None or val == '':
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False  # Can't parse = real text
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get('text', '')
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get('text', '').strip() if isinstance(item, dict) and item.get('type') == 'text'
            else (item.strip() if isinstance(item, str) else '')
            for item in content
        ]
        result = ' '.join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """Classify LangChain message into display type and extract content."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, 'content', None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # Fallback for unknown types
    return ("System", content)


def run_analysis_simple(selections):
    """Run simplified analysis without UI, returning final_state.

    使用固定配置运行分析。

    Args:
        selections: Dict with configuration values (ticker, analysis_date, research_depth, debug)

    Returns:
        final_state: Dict with complete analysis results
    """
    from langchain_openai import ChatOpenAI
    from tradingagents.graph.trading_graph_simple import SIMPLE_CONFIG

    # 创建固定配置的 LLM 实例
    llm = ChatOpenAI(
        model="Qwen3-14B",
        api_key="your-api-key-here",  # 从环境变量读取或直接设置
        base_url="https://u701950-942d-62b9bc31.bjb1.seetacloud.com:8443/v1",
        max_tokens=30000,
        temperature=0.1,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    # Create config by merging with SIMPLE_CONFIG defaults
    config = SIMPLE_CONFIG.copy()
    config["max_debate_rounds"] = selections.get("research_depth", 1)
    config["max_risk_discuss_rounds"] = selections.get("research_depth", 1)

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # 注意：trading_graph_simple.py 使用固定的分析师集合（所有4个）
    selected_analyst_keys = ["market", "social", "news", "fundamentals"]

    # Initialize the graph with fixed LLM
    graph = TradingAgentsGraph(
        debug=selections.get("debug", False),
        config=config,
        llm=llm,  # 传入固定配置的 LLM
    )

    # Initialize message buffer
    message_buffer = MessageBuffer()
    message_buffer.init_for_analysis(selected_analyst_keys)

    # Create result directory
    results_dir = Path("./results") / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Add initial messages
    message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
    message_buffer.add_message("System", f"Analysis date: {selections['analysis_date']}")
    message_buffer.add_message(
        "System",
        f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
    )

    # Update agent status to in_progress for the first analyst
    first_analyst = f"{selections['analysts'][0].value.capitalize()} Analyst"
    message_buffer.update_agent_status(first_analyst, "in_progress")

    # Initialize state and get graph args with callbacks
    init_agent_state = graph.propagator.create_initial_state(
        selections["ticker"], selections["analysis_date"]
    )
    args = graph.propagator.get_graph_args(callbacks=[stats_handler])

    # Stream the analysis
    trace = []
    for chunk in graph.graph.stream(init_agent_state, **args):
        # Process messages if present (skip duplicates via message ID)
        if len(chunk["messages"]) > 0:
            last_message = chunk["messages"][-1]
            msg_id = getattr(last_message, "id", None)

            if msg_id != message_buffer._last_message_id:
                message_buffer._last_message_id = msg_id

                # Add message to buffer
                msg_type, content = classify_message_type(last_message)
                if content and content.strip():
                    message_buffer.add_message(msg_type, content)

                # Handle tool calls
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(
                                tool_call["name"], tool_call["args"]
                            )
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

        # Update analyst statuses based on report state
        update_analyst_statuses(message_buffer, chunk)

        # Research Team - Handle Investment Debate State
        if chunk.get("investment_debate_state"):
            debate_state = chunk["investment_debate_state"]
            bull_hist = debate_state.get("bull_history", "").strip()
            bear_hist = debate_state.get("bear_history", "").strip()
            judge = debate_state.get("judge_decision", "").strip()

            # Only update status when there's actual content
            if bull_hist or bear_hist:
                for agent in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                    message_buffer.update_agent_status(agent, "in_progress")
            if bull_hist:
                message_buffer.update_report_section(
                    "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                )
            if bear_hist:
                message_buffer.update_report_section(
                    "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                )
            if judge:
                message_buffer.update_report_section(
                    "investment_plan", f"### Research Manager Decision\n{judge}"
                )
                for agent in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                    message_buffer.update_agent_status(agent, "completed")
                message_buffer.update_agent_status("Trader", "in_progress")

        # Trading Team
        if chunk.get("trader_investment_plan"):
            message_buffer.update_report_section(
                "trader_investment_plan", chunk["trader_investment_plan"]
            )
            if message_buffer.agent_status.get("Trader") != "completed":
                message_buffer.update_agent_status("Trader", "completed")
                for agent in ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"]:
                    message_buffer.update_agent_status(agent, "in_progress")

        # Risk Management Team - Handle Risk Debate State
        if chunk.get("risk_debate_state"):
            risk_state = chunk["risk_debate_state"]
            agg_hist = risk_state.get("aggressive_history", "").strip()
            con_hist = risk_state.get("conservative_history", "").strip()
            neu_hist = risk_state.get("neutral_history", "").strip()
            judge = risk_state.get("judge_decision", "").strip()

            if agg_hist:
                if message_buffer.agent_status.get("Aggressive Analyst") != "completed":
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                message_buffer.update_report_section(
                    "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}"
                )
            if con_hist:
                if message_buffer.agent_status.get("Conservative Analyst") != "completed":
                    message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                message_buffer.update_report_section(
                    "final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}"
                )
            if neu_hist:
                if message_buffer.agent_status.get("Neutral Analyst") != "completed":
                    message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                message_buffer.update_report_section(
                    "final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}"
                )
            if judge:
                if message_buffer.agent_status.get("Portfolio Manager") != "completed":
                    message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                    )
                    message_buffer.update_agent_status("Aggressive Analyst", "completed")
                    message_buffer.update_agent_status("Conservative Analyst", "completed")
                    message_buffer.update_agent_status("Neutral Analyst", "completed")
                    message_buffer.update_agent_status("Portfolio Manager", "completed")

        trace.append(chunk)

    # Get final state and decision
    final_state = trace[-1]
    decision = graph.process_signal(final_state["final_trade_decision"])

    # Update all agent statuses to completed
    for agent in message_buffer.agent_status:
        message_buffer.update_agent_status(agent, "completed")

    message_buffer.add_message(
        "System", f"Completed analysis for {selections['analysis_date']}"
    )

    # Update final report sections
    for section in message_buffer.report_sections.keys():
        if section in final_state:
            message_buffer.update_report_section(section, final_state[section])

    return final_state


def get_default_selections():
    """Return default configuration values."""
    return DEFAULT_CONFIG_VALUES.copy()


def main():
    """Run simplified analysis with default configuration.

    Usage: python cli/main_simple.py
    """
    # Get default selections
    selections = get_default_selections()

    # Run analysis and return final state
    final_state = run_analysis_simple(selections)

    # Return final state for caller to use
    return final_state


if __name__ == "__main__":
    main()
